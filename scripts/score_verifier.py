"""Verifier scores for every evaluation set (artifacts/scores/verifier_<set>.parquet: ft_sup, ft_nei, ft_con).

Train and the augmented train rows get the out-of-fold scores written by scripts/train_verifier.py; every other set is
scored by the average of the three fold models, exactly as the service does.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate import quantities  # noqa: E402
from claimgate.data import ROOT  # noqa: E402
from claimgate.features import LIST_ITEM  # noqa: E402
from claimgate.text import context_for_nli, sentences  # noqa: E402
from claimgate.verifier import Verifier, premise  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_detectors import load_set  # noqa: E402

SETS = ["dev", "test", "quantstress", "ragtruth_dev300", "ragtruth_test"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=SETS)
    args = ap.parse_args()
    out_dir = ROOT / "artifacts" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "artifacts" / "verifier").glob("oof_fold*.parquet"))])
    oof[~oof.uid.str.startswith("aug")].to_parquet(out_dir / "verifier_train.parquet")
    oof[oof.uid.str.startswith("aug")].to_parquet(out_dir / "verifier_train_aug.parquet")
    print(f"train / train_aug: {len(oof)} out-of-fold rows", flush=True)
    model = Verifier()
    for name in args.sets:
        t0 = time.time()
        rows, pairs = [], []
        for ex in load_set(name):
            ctx = context_for_nli(ex)
            cs = sentences(ctx)
            for j, s in enumerate(ex.sentences):
                hyp = quantities.canonicalize(LIST_ITEM.sub("", s).strip() or s, ctx)[0]
                rows.append((ex.uid, j))
                pairs.append((premise(cs, hyp), hyp))
        p = model.score(pairs)
        pd.DataFrame(rows, columns=["uid", "sent_idx"]).assign(ft_sup=p[:, 0], ft_nei=p[:, 1], ft_con=p[:, 2]).to_parquet(out_dir / f"verifier_{name}.parquet")
        print(f"{name}: {len(pairs)} claims, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
