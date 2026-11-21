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
from claimgate.verifier import VERIFIER_DIR, Verifier, premise  # noqa: E402

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
    print(f"train: {int((~oof.uid.str.startswith('aug')).sum())} out-of-fold rows", flush=True)
    # rule-made rows: score each with the fold model that never saw its source answer (robust to the edit list changing)
    import random

    from claimgate.data import load_delucionqa
    from claimgate.stress import train_augmentation
    uids = sorted(e.uid for e in load_delucionqa("train"))
    random.Random(123).shuffle(uids)
    fold_of = {u: i % 3 for i, u in enumerate(uids)}
    aug = train_augmentation()
    rows = []
    for k in range(3):
        fold_model = Verifier(dirs=[VERIFIER_DIR / f"fold{k}"])
        part = [e for e in aug if fold_of[e.meta["source_uid"]] == k]
        pairs = []
        for e in part:
            ctx = context_for_nli(e)
            hyp = quantities.canonicalize(LIST_ITEM.sub("", e.sentences[0]).strip() or e.sentences[0], ctx)[0]
            pairs.append((premise(sentences(ctx), hyp), hyp))
        p = fold_model.score(pairs)
        rows += [(e.uid, 0, *x) for e, x in zip(part, p, strict=True)]
        del fold_model
    pd.DataFrame(rows, columns=["uid", "sent_idx", "ft_sup", "ft_nei", "ft_con"]).to_parquet(out_dir / "verifier_train_aug.parquet")
    print(f"train_aug: {len(rows)} rows scored by the fold model that held out their source answer", flush=True)
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
