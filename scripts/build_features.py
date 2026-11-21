"""Compute per-sentence evidence features for DelucionQA train / dev / test (NLI results are cached in artifacts/)."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.features import sentence_features  # noqa: E402
from claimgate.nli import NLI  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    ap.add_argument("--model", default=str(ROOT / "models" / "deberta-v3-large-nli"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    nli = NLI(args.model, batch_size=args.batch_size)
    out = ROOT / "artifacts" / "features"
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).name
    for split in args.splits:
        t0 = time.time()
        df = sentence_features(load_delucionqa(split), nli)
        df.to_parquet(out / f"{split}_{tag}.parquet")
        print(f"{split}: {len(df)} sentences, {df.uid.nunique()} answers, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
