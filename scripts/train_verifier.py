"""Cross-fitted fine-tuning of the claim verifier on DelucionQA train (+ rule-made contradictions of train sentences).

3 folds over train answers. Each fold model is trained on the other two folds and scores its own fold, so every train
sentence (and every augmented row, assigned to the fold of the answer it came from) gets an out-of-fold score for the
decision layer. Dev and test are never used here; the three fold models are averaged at inference (claimgate.verifier).

Stable on a 16 GB Apple M5: float32 master weights with float16 autocast (the checkpoint is stored in float16, and
training it in float16 makes AdamW divide by underflowed second moments), embeddings and the lower 16 of 24 layers
frozen, batch 4 x 4 accumulation, premises limited to the claim's best evidence windows.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate import quantities  # noqa: E402
from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.features import LIST_ITEM  # noqa: E402
from claimgate.nli import device  # noqa: E402
from claimgate.stress import train_augmentation  # noqa: E402
from claimgate.text import context_for_nli, sentences  # noqa: E402
from claimgate.verifier import MAX_LEN, VERIFIER_DIR, premise  # noqa: E402

LABEL = {"Supported": 0, "Neither": 1, "Conflicted": 2}


def claim_rows(examples, fold_of=None):
    rows = []
    for ex in examples:
        ctx = context_for_nli(ex)
        cs = sentences(ctx)
        src = ex.meta.get("source_uid", ex.uid)
        for j, (s, lab) in enumerate(zip(ex.sentences, ex.sentence_labels, strict=True)):
            hyp = quantities.canonicalize(LIST_ITEM.sub("", s).strip() or s, ctx)[0]
            rows.append({"uid": ex.uid, "sent_idx": j, "premise": premise(cs, hyp), "hyp": hyp, "y": LABEL[lab],
                         "fold": fold_of[src] if fold_of else -1})
    return rows


def batches(rows, bs, tok, shuffle, seed=0):
    idx = list(range(len(rows)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
        idx = [i for b in range(0, len(idx), bs * 32) for i in sorted(idx[b:b + bs * 32], key=lambda i: len(rows[i]["premise"]))]
        groups = [idx[b:b + bs] for b in range(0, len(idx), bs)]
        rnd.shuffle(groups)
    else:
        idx.sort(key=lambda i: len(rows[i]["premise"]))
        groups = [idx[b:b + bs] for b in range(0, len(idx), bs)]
    for g in groups:
        enc = tok([rows[i]["premise"] for i in g], [rows[i]["hyp"] for i in g], truncation="only_first", max_length=MAX_LEN, padding=True,
                  return_tensors="pt")
        yield g, enc, torch.tensor([rows[i]["y"] for i in g])


@torch.inference_mode()
def predict(model, tok, rows, bs=16):
    model.eval()
    out = np.zeros((len(rows), 3), dtype=np.float32)
    for g, enc, _ in batches(rows, bs, tok, shuffle=False):
        with torch.autocast("mps", dtype=torch.float16, enabled=model.device.type == "mps"):
            logits = model(**enc.to(model.device)).logits
        out[g] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def train_fold(args, train_rows, tok, log):
    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, dtype=torch.float32).to(device())
    for p in model.deberta.embeddings.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.deberta.encoder.layer):
        if i < args.freeze:
            for p in layer.parameters():
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    counts = np.bincount([r["y"] for r in train_rows], minlength=3)
    weight = torch.tensor(np.clip(counts.sum() / (3 * np.maximum(counts, 1)), 0.5, 4.0), dtype=torch.float32, device=model.device)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train_rows) / args.bs / args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (_, enc, y) in enumerate(batches(train_rows, args.bs, tok, shuffle=True, seed=args.seed + epoch)):
            with torch.autocast("mps", dtype=torch.float16, enabled=model.device.type == "mps"):
                logits = model(**enc.to(model.device)).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y.to(model.device), weight=weight) / args.accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")
            loss.backward()
            total += loss.item() * args.accum
            n += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step} loss {total / n:.4f} {time.time() - t0:.0f}s", flush=True)
        log.append({"epoch": epoch, "train_loss": total / n, "secs": time.time() - t0})
        print(f"  epoch {epoch} done: loss {total / n:.4f}, {time.time() - t0:.0f}s", flush=True)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "models" / "deberta-v3-large-nli"))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--only-fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = load_delucionqa("train")
    uids = sorted(e.uid for e in train)
    random.Random(123).shuffle(uids)
    fold_of = {u: i % args.folds for i, u in enumerate(uids)}
    rows = claim_rows(train, fold_of) + claim_rows(train_augmentation(), fold_of)
    tok = AutoTokenizer.from_pretrained(args.base)
    out_dir = ROOT / "artifacts" / "verifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} training rows ({sum(r['uid'].startswith('aug') for r in rows)} augmented), {args.folds} folds", flush=True)
    for k in range(args.folds):
        if args.only_fold is not None and k != args.only_fold:
            continue
        tr = [r for r in rows if r["fold"] != k]
        ho = [r for r in rows if r["fold"] == k]
        print(f"fold {k}: train {len(tr)}, held out {len(ho)}", flush=True)
        log = []
        model = train_fold(args, tr, tok, log)
        p = predict(model, tok, ho)
        pd.DataFrame({"uid": [r["uid"] for r in ho], "sent_idx": [r["sent_idx"] for r in ho], "ft_sup": p[:, 0], "ft_nei": p[:, 1],
                      "ft_con": p[:, 2]}).to_parquet(out_dir / f"oof_fold{k}.parquet")
        dest = VERIFIER_DIR / f"fold{k}"
        model.half().save_pretrained(dest)
        tok.save_pretrained(dest)
        y = np.array([r["y"] for r in ho])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y != 0, 1 - p[:, 0])
        (out_dir / f"fold{k}_log.json").write_text(json.dumps({"args": vars(args), "log": log, "oof_auc_unsupported": auc}, indent=1))
        print(f"fold {k}: out-of-fold AUC (unsupported vs supported) {auc:.4f}", flush=True)
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
