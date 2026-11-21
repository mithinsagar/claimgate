"""Re-implementations of the DelucionQA paper's similarity baselines (Sadat et al., 2023, Section 4.1).

Sim-overlap: an answer sentence is grounded if, for some context sentence, the overlap ratio exceeds a threshold. Overlap
length counts tokens of aligned phrases with at least min(4, 30% of the shorter sentence) tokens; ratio = overlap / shorter
length. An answer is Hallucinated if any sentence is not grounded. The threshold is tuned on dev over {0.1, ..., 0.9}, as in
the paper. The paper's wording ("more than either 4 or n words") allows several readings; this one reproduces the
published test macro F1 (71.00 here vs 71.09 published) on the reconstructed contexts, which also validates them.
"""
from __future__ import annotations

import difflib
import re

from nltk import word_tokenize

from claimgate.data import Example

WL = set("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~'")


def bosch_tokens(text: str) -> list[str]:
    text = "".join(filter(WL.__contains__, text.lower()))
    text = re.sub(r"\.(?!(?<=\d\.)\d) ?", ". ", text)
    return word_tokenize(text)


def overlap_ratio(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    shorter = min(len(a), len(b))
    min_len = min(4, 0.3 * shorter)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    overlap = sum(m.size for m in sm.get_matching_blocks() if m.size >= min_len)
    return overlap / shorter


def context_token_sentences(ex: Example) -> list[list[str]]:
    toks = ex.context.split()
    sents, cur = [], []
    for t in toks:
        cur.append(t)
        if t in {".", "!", "?"}:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def sim_overlap_sentence_scores(ex: Example) -> list[float]:
    """Max overlap ratio of each answer sentence against any context sentence (higher = more grounded)."""
    ctx = context_token_sentences(ex)
    return [max((overlap_ratio(bosch_tokens(s), c) for c in ctx), default=0.0) for s in ex.sentences]


def sim_overlap_predict(examples: list[Example], threshold: float) -> list[int]:
    return [int(any(v <= threshold for v in sim_overlap_sentence_scores(ex))) for ex in examples]
