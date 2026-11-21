"""Fine-tune the NLI cross-encoder on DelucionQA train sentences (context, answer sentence) -> Supported / Neither / Conflicted.

The 3-way NLI head is reused: Supported -> entailment, Neither -> neutral, Conflicted -> contradiction, so training
starts from a model that already knows the task shape. Embeddings are frozen and gradient checkpointing is on so that
DeBERTa-v3-large trains within 16 GB of unified memory. The epoch is chosen by dev answer-level macro F1; test is only
scored with the chosen checkpoint. Optional --folds/--fold produce out-of-fold train scores for stacking.
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

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.decide import tune_threshold  # noqa: E402
from claimgate.features import LIST_ITEM  # noqa: E402
from claimgate.nli import device  # noqa: E402
from claimgate.text import context_for_nli  # noqa: E402

LABEL = {"Supported": 0, "Neither": 1, "Conflicted": 2}


def rows(examples):
    out = []
    for ex in examples:
        ctx = context_for_nli(ex)
        for j, (s, lab) in enumerate(zip(ex.sentences, ex.sentence_labels, strict=True)):
            out.append({"uid": ex.uid, "sent_idx": j, "premise": ctx, "hyp": LIST_ITEM.sub("", s).strip() or s, "y": LABEL[lab],
                        "answer_label": ex.label})
    return out


def batches(data, bs, tok, max_len, shuffle, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(idx)
        # length-grouped buckets keep padding low
        idx = [i for b in range(0, len(idx), bs * 32) for i in sorted(idx[b:b + bs * 32], key=lambda i: len(data[i]["premise"]))]
        groups = [idx[b:b + bs] for b in range(0, len(idx), bs)]
        rnd.shuffle(groups)
    else:
        idx.sort(key=lambda i: len(data[i]["premise"]))
        groups = [idx[b:b + bs] for b in range(0, len(idx), bs)]
    for g in groups:
        enc = tok([data[i]["premise"] for i in g], [data[i]["hyp"] for i in g], truncation="only_first", max_length=max_len, padding=True,
                  return_tensors="pt")
        yield g, enc, torch.tensor([data[i]["y"] for i in g])


@torch.inference_mode()
def predict(model, tok, data, bs, max_len):
    model.eval()
    probs = np.zeros((len(data), 3), dtype=np.float32)
    for g, enc, _ in batches(data, bs, tok, max_len, shuffle=False):
        logits = model(**enc.to(model.device)).logits.float()
        probs[g] = torch.softmax(logits, -1).cpu().numpy()
    return probs


def answer_f1(data, risk):
    d = pd.DataFrame({"uid": [r["uid"] for r in data], "y": [r["answer_label"] for r in data], "s": risk})
    a = d.groupby("uid").agg(y=("y", "first"), s=("s", "max"))
    return tune_threshold(a.y.to_numpy(), a.s.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "deberta-v3-large-nli"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "verifier"))
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train, dev, test = rows(load_delucionqa("train")), rows(load_delucionqa("dev")), rows(load_delucionqa("test"))
    held = []
    tag = "full"
    if args.folds:
        uids = sorted({r["uid"] for r in train})
        random.Random(123).shuffle(uids)
        fold_uids = set(uids[args.fold::args.folds])
        held = [r for r in train if r["uid"] in fold_uids]
        train = [r for r in train if r["uid"] not in fold_uids]
        tag = f"fold{args.fold}of{args.folds}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device())
    model.gradient_checkpointing_enable()
    for p in model.deberta.embeddings.parameters():
        p.requires_grad = False
    counts = np.bincount([r["y"] for r in train], minlength=3)
    weights = torch.tensor(np.clip(counts.sum() / (3 * np.maximum(counts, 1)), 0.5, 4.0), dtype=torch.float32, device=model.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / args.bs / args.accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    print(f"train {len(train)} sentences, dev {len(dev)}, test {len(test)}, held {len(held)}, class weights {weights.tolist()}", flush=True)

    best, log = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for step, (_, enc, y) in enumerate(batches(train, args.bs, tok, args.max_len, shuffle=True, seed=args.seed + epoch)):
            loss = loss_fn(model(**enc.to(model.device)).logits.float(), y.to(model.device)) / args.accum
            loss.backward()
            total += loss.item() * args.accum
            n += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {total / n:.4f} {time.time() - t0:.0f}s", flush=True)
        dprob = predict(model, tok, dev, 16, args.max_len)
        t, f = answer_f1(dev, 1 - dprob[:, 0])
        log.append({"epoch": epoch, "train_loss": total / n, "dev_answer_macro_f1": f, "threshold": t, "secs": time.time() - t0})
        print(log[-1], flush=True)
        if f > best:
            best = f
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
            for name, data in (("dev", dev), ("test", test), ("held", held)):
                if not data:
                    continue
                p = dprob if name == "dev" else predict(model, tok, data, 16, args.max_len)
                pd.DataFrame({"uid": [r["uid"] for r in data], "sent_idx": [r["sent_idx"] for r in data], "p_sup": p[:, 0], "p_nei": p[:, 1],
                              "p_con": p[:, 2]}).to_parquet(out / f"{name}_scores.parquet")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log, "best_dev_answer_macro_f1": best}, indent=1))


if __name__ == "__main__":
    main()
