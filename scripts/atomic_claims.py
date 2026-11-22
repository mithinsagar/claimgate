"""Atomic claims: split compound answer sentences into self-contained facts with the local LLM and score each part.

Sentence risk becomes the riskiest of (the whole sentence, its parts). Decision rule fixed before running: adopt only if
dev macro F1 improves by at least 0.5 points; test is processed only in that case (otherwise only dev is reported).
Writes artifacts/atomic/<split>.jsonl (parts) and results/atomic_claims.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, Example, load_delucionqa  # noqa: E402
from claimgate.decide import answer_scores, contradiction_gate, tune_threshold  # noqa: E402
from claimgate.metrics import macro_f1  # noqa: E402
from claimgate.pipeline import FULL_MODEL_PATH  # noqa: E402

COMPOUND = re.compile(r"\b(and|but|while|which|whereas|so that|as well as)\b|;|,.*,")
PROMPT = ("Split the sentence into short, self-contained factual claims. Keep the original wording where possible and "
          "replace pronouns with what they refer to. If it states only one fact, return it unchanged.\n"
          "Return only a JSON list of strings.\n\nSentence: {s}")
OUT = ROOT / "artifacts" / "atomic"


def is_compound(s: str) -> bool:
    return len(s.split()) >= 15 and bool(COMPOUND.search(s))


def split_sentences(split: str) -> list[dict]:
    path = OUT / f"{split}.jsonl"
    done = {(r["uid"], r["sent_idx"]) for r in map(json.loads, open(path))} if path.exists() else set()
    todo = [(e, j, s) for e in load_delucionqa(split) for j, s in enumerate(e.sentences) if is_compound(s) and (e.uid, j) not in done]
    if todo:
        import mlx.core as mx
        from mlx_lm import generate, load
        mx.set_cache_limit(1 << 30)
        model, tok = load(str(ROOT / "models" / "qwen2.5-7b-instruct-4bit"))
        t0 = time.time()
        with open(path, "a") as f:
            for k, (e, j, s) in enumerate(todo):
                prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(s=s)}], add_generation_prompt=True, tokenize=False)
                text = generate(model, tok, prompt=prompt, max_tokens=200, verbose=False)
                try:
                    parts = [p.strip() for p in json.loads(text[text.index("["): text.rindex("]") + 1]) if isinstance(p, str) and p.strip()]
                except ValueError:
                    parts = []
                f.write(json.dumps({"uid": e.uid, "sent_idx": j, "parts": parts[:5] if len(parts) > 1 else []}) + "\n")
                f.flush()
                if k % 100 == 0:
                    print(f"{split} {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                mx.clear_cache()
    return [r for r in map(json.loads, open(path)) if r["parts"]]


def part_features(split: str, rows: list[dict]) -> pd.DataFrame:
    """Features for every part, keyed back to (uid, sent_idx) of the sentence it came from."""
    path = ROOT / "artifacts" / "features" / f"atomic_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from claimgate.detectors import HHEM
    from claimgate.features import sentence_features
    from claimgate.nli import NLI
    by_uid = {e.uid: e for e in load_delucionqa(split)}
    exs, owner = [], []
    for r in rows:
        e = by_uid[r["uid"]]
        exs.append(Example(uid=f"{r['uid']}#{r['sent_idx']}", split=split, question=e.question, context=e.context, context_cased=e.context_cased,
                           answer=e.answer, sentences=r["parts"], sentence_labels=["Supported"] * len(r["parts"]), label=0,
                           coverage=e.coverage, meta=dict(e.meta)))
        owner.append((r["uid"], r["sent_idx"]))
    df = sentence_features(exs, NLI(verbose=True))
    from claimgate.text import context_for_nli
    df["hhem"] = HHEM().score([(context_for_nli(e), p) for e in exs for p in e.sentences])
    df["src_uid"] = df.uid.str.split("#").str[0]
    df["src_sent"] = df.uid.str.split("#").str[1].astype(int)
    df.to_parquet(path)
    return df


def evaluate(split: str, g: dict, use_gate: bool) -> dict:
    base = pd.read_parquet(ROOT / "artifacts" / "features" / f"{split}_deberta-v3-large-nli.parquet")
    from claimgate.signals import load_scores
    base = load_scores(split, base)
    risk = g["model"].predict_proba(base)
    if use_gate:
        risk = np.where(contradiction_gate(base, g.get("contra")), 1.0, risk)
    parts = part_features(split, split_sentences(split))
    p_risk = g["model"].predict_proba(parts)
    if use_gate:
        p_risk = np.where(contradiction_gate(parts, g.get("contra")), 1.0, p_risk)
    worst = pd.Series(p_risk, index=pd.MultiIndex.from_arrays([parts.src_uid, parts.src_sent])).groupby(level=[0, 1]).max()
    key = pd.MultiIndex.from_arrays([base.uid, base.sent_idx])
    combined = np.maximum(risk, worst.reindex(key).fillna(0).to_numpy())
    return {"sentence": risk, "atomic": combined, "frame": base, "n_split": int(parts[["src_uid", "src_sent"]].drop_duplicates().shape[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g = joblib.load(FULL_MODEL_PATH)
    res = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    use_gate = res["design_decisions"].get("use_gate", True)
    dev = evaluate("dev", g, use_gate)
    out = {"n_dev_sentences_split": dev["n_split"]}
    for name in ("sentence", "atomic"):
        a = answer_scores(dev["frame"], dev[name])
        t, f = tune_threshold(a.label.to_numpy(), a.score.to_numpy())
        out[f"dev_{name}"] = {"threshold": t, "macro_f1": f}
    out["adopt"] = out["dev_atomic"]["macro_f1"] >= out["dev_sentence"]["macro_f1"] + 0.005
    print({k: v for k, v in out.items()}, flush=True)
    if out["adopt"] and args.stage == "test":
        test = evaluate("test", g, use_gate)
        for name in ("sentence", "atomic"):
            a = answer_scores(test["frame"], test[name])
            out[f"test_{name}"] = macro_f1(a.label, a.score >= out[f"dev_{name}"]["threshold"])
        print({k: v for k, v in out.items() if k.startswith("test")}, flush=True)
    (ROOT / "results" / "atomic_claims.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
