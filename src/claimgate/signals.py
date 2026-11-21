"""Extra per-claim signals beyond the zero-shot NLI features: the fine-tuned verifier and the open detectors.

For evaluation they are read from artifacts (scored once by scripts/train_verifier.py, scripts/score_verifier.py and
scripts/score_detectors.py); for live requests they are computed on the fly by the scorers passed to the pipeline.
"""
from __future__ import annotations

import pandas as pd

from claimgate import quantities
from claimgate.data import ROOT, Example
from claimgate.features import LIST_ITEM
from claimgate.text import context_for_nli, sentences

SIGNAL_COLUMNS = {"ft_sup": "verifier", "ft_con": "verifier", "hhem": "hhem", "minicheck": "minicheck"}
SCORES = ROOT / "artifacts" / "scores"


def add_signals(df: pd.DataFrame, examples: list[Example], scorers: dict, needed: set[str]) -> pd.DataFrame:
    """Compute the needed signals live for the claims of `examples` (in df row order)."""
    if not needed:
        return df
    df = df.copy()
    claims = [(ex, LIST_ITEM.sub("", s).strip() or s) for ex in examples for s in ex.sentences]
    if "verifier" in needed:
        from claimgate.verifier import premise
        pairs = []
        for ex, c in claims:
            ctx = context_for_nli(ex)
            hyp = quantities.canonicalize(c, ctx)[0]
            pairs.append((premise(sentences(ctx), hyp), hyp))
        p = scorers["verifier"].score(pairs)
        df["ft_sup"], df["ft_con"] = p[:, 0], p[:, 2]
    for name in ("hhem", "minicheck"):
        if name in needed:
            df[name] = scorers[name].score([(context_for_nli(ex), c) for ex, c in claims])
    return df


def load_scores(split: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach every precomputed signal available for `split` (artifacts/scores/<signal>_<split>.parquet)."""
    out = frame
    for path in sorted(SCORES.glob(f"*_{split}.parquet")):
        name = path.stem[: -len(split) - 1]
        if name not in ("verifier", "hhem", "minicheck"):
            continue
        sig = pd.read_parquet(path).drop_duplicates(["uid", "sent_idx"])
        out = out.merge(sig, on=["uid", "sent_idx"], how="left")
    return out
