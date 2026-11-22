"""Score every claim of every evaluation set with HHEM-2.1-Open or MiniCheck (artifacts/scores/<detector>_<set>.parquet).

The premise is the full evidence (DelucionQA context or RAGTruth passages); the hypothesis is the claim sentence.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.features import LIST_ITEM  # noqa: E402
from claimgate.ragtruth import load_ragtruth_qa  # noqa: E402
from claimgate.stress import as_example, generate, train_augmentation  # noqa: E402
from claimgate.text import context_for_nli  # noqa: E402

SETS = ["dev", "test", "train", "train_aug", "quantstress", "ragtruth_dev300", "ragtruth_test", "ragtruth_train600"]


def load_set(name: str):
    if name in ("train", "dev", "test"):
        return load_delucionqa(name)
    if name == "train_aug":
        return train_augmentation()
    if name == "quantstress":
        return [as_example(it, ex, i) for i, (it, ex) in enumerate(generate(load_delucionqa("dev") + load_delucionqa("test")))]
    if name == "ragtruth_dev300":
        return load_ragtruth_qa("train", limit=300)
    if name == "ragtruth_test":
        return load_ragtruth_qa("test")
    if name in ("ragtruth_train600", "ragtruth_train_all"):
        from claimgate.ragtruth import adaptation_split
        return adaptation_split(600 if name == "ragtruth_train600" else None)[0]
    raise KeyError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("detector", choices=["hhem", "minicheck"])
    ap.add_argument("--sets", nargs="+", default=SETS)
    args = ap.parse_args()
    from claimgate.detectors import HHEM, MiniCheck
    model = HHEM() if args.detector == "hhem" else MiniCheck()
    out_dir = ROOT / "artifacts" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        path = out_dir / f"{args.detector}_{name}.parquet"
        if path.exists():
            print(f"{name}: cached", flush=True)
            continue
        t0 = time.time()
        rows, pairs = [], []
        for ex in load_set(name):
            evidence = context_for_nli(ex)
            for j, s in enumerate(ex.sentences):
                rows.append((ex.uid, j))
                pairs.append((evidence, LIST_ITEM.sub("", s).strip() or s))
        scores = model.score(pairs)
        pd.DataFrame(rows, columns=["uid", "sent_idx"]).assign(**{args.detector: scores}).to_parquet(path)
        print(f"{name}: {len(pairs)} claims, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
