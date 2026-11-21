"""Per-sentence evidence features: NLI over context windows, lexical grounding, quantity checks and surface cues.

Every answer sentence is a claim. For each one we keep the best-supporting evidence window, its NLI probabilities, how
much of the sentence's content vocabulary is absent from the evidence, and the deterministic quantity verdicts. The
learned decision layer (decide.py) only sees these numbers, which keeps every verdict traceable to evidence.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from claimgate import quantities
from claimgate.baselines import bosch_tokens, context_token_sentences, overlap_ratio
from claimgate.data import Example
from claimgate.text import content_words, context_for_nli, sentences, stem

HEDGES = re.compile(r"\b(usually|typically|generally|often|commonly|may|might|can be|recommend\w*|consult\w*|qualified|professional|"
                    r"important|always|never|ensure|make sure|be sure|safety|safe|tips?|best practice)\b", re.I)
REFUSAL = re.compile(r"(not (mentioned|provided|specified|included|stated)|does not (provide|mention|specify|contain|say)|"
                     r"no (information|mention)|i'?m sorry|i am sorry|unable to|cannot (provide|answer)|given context)", re.I)
LIST_ITEM = re.compile(r"^\s*(\d+[.)]|[-•*])\s*")


def windows(ctx_sents: list[str], size: int = 2) -> list[str]:
    if len(ctx_sents) <= size:
        return [" ".join(ctx_sents)] if ctx_sents else [""]
    return [" ".join(ctx_sents[i:i + size]) for i in range(len(ctx_sents) - size + 1)]


def chunks(ctx_sents: list[str], max_words: int = 280) -> list[str]:
    out, cur, n = [], [], 0
    for s in ctx_sents:
        w = len(s.split())
        if cur and n + w > max_words:
            out.append(" ".join(cur))
            cur, n = cur[len(cur) // 2:], sum(len(x.split()) for x in cur[len(cur) // 2:])
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out


def _top_windows(hyp: str, wins: list[str], k: int) -> list[int]:
    hw = {stem(w) for w in content_words(hyp)}
    scores = [len(hw & {stem(w) for w in content_words(win)}) for win in wins]
    return sorted(range(len(wins)), key=lambda i: -scores[i])[:k]


def build_pairs(ex: Example, top_k: int = 6, canonical: bool = True) -> tuple[list[tuple[str, str]], list[list[int]], list[list[int]]]:
    """NLI pairs for every answer sentence; returns pairs plus per-sentence indices of window and chunk pairs."""
    ctx = context_for_nli(ex)
    cs = sentences(ctx)
    wins, chs = windows(cs), chunks(cs)
    pairs, win_idx, ch_idx = [], [], []
    for s in ex.sentences:
        hyp = LIST_ITEM.sub("", s).strip() or s
        if canonical:
            hyp, _ = quantities.canonicalize(hyp, ctx)
        wi = []
        for i in _top_windows(hyp, wins, top_k):
            wi.append(len(pairs))
            pairs.append((wins[i], hyp))
        ci = []
        for c in chs:
            ci.append(len(pairs))
            pairs.append((c, hyp))
        win_idx.append(wi)
        ch_idx.append(ci)
    return pairs, win_idx, ch_idx


def sentence_features(examples: list[Example], nli, canonical: bool = True) -> pd.DataFrame:
    all_pairs, plan = [], []
    for ex in examples:
        pairs, wi, ci = build_pairs(ex, canonical=canonical)
        off = len(all_pairs)
        all_pairs.extend(pairs)
        plan.append(([[off + i for i in w] for w in wi], [[off + i for i in c] for c in ci]))
    probs = nli.score(all_pairs) if all_pairs else np.zeros((0, 3))

    rows = []
    for ex, (wi, ci) in zip(examples, plan, strict=True):
        ctx_lower = ex.context
        ctx_stems = {stem(w) for w in content_words(ctx_lower)} | {stem(w) for w in content_words(ex.question)}
        ctx_toks = context_token_sentences(ex)
        evidence = context_for_nli(ex)
        n = len(ex.sentences)
        for j, s in enumerate(ex.sentences):
            pw, pc = probs[wi[j]], probs[ci[j]]
            best = int(np.argmax(pw[:, 0]))
            cw = [stem(w) for w in content_words(s)]
            novel = [w for w in cw if w not in ctx_stems]
            q = quantities.features(s, evidence)
            st = bosch_tokens(s)
            rows.append({
                "uid": ex.uid, "split": ex.split, "sent_idx": j, "n_sent": n, "rel_pos": j / max(1, n - 1),
                "target": ex.sentence_targets[j], "answer_label": ex.label, "coverage": ex.coverage,
                "e_win_max": float(pw[:, 0].max()), "c_at_best": float(pw[best, 2]), "n_at_best": float(pw[best, 1]),
                "c_win_max": float(pw[:, 2].max()), "e_win_mean": float(pw[:, 0].mean()),
                "e_chunk_max": float(pc[:, 0].max()), "c_chunk_max": float(pc[:, 2].max()), "n_chunk_min": float(pc[:, 1].min()),
                "overlap": max((overlap_ratio(st, c) for c in ctx_toks), default=0.0),
                "novel_frac": len(novel) / max(1, len(cw)), "novel_n": len(novel), "n_content": len(cw), "n_words": len(s.split()),
                "hedge": len(HEDGES.findall(s)), "refusal": int(bool(REFUSAL.search(s))), "list_item": int(bool(LIST_ITEM.match(s))),
                "yes_no": int(bool(re.match(r"^(yes|no)\b", s.strip(), re.I))), **q,
                "best_evidence": all_pairs[wi[j][best]][0], "claim_for_nli": all_pairs[wi[j][best]][1],
            })
    return pd.DataFrame(rows)


FEATURES = ["sent_idx", "n_sent", "rel_pos", "e_win_max", "c_at_best", "n_at_best", "c_win_max", "e_win_mean", "e_chunk_max",
            "c_chunk_max", "n_chunk_min", "overlap", "novel_frac", "novel_n", "n_content", "n_words", "hedge", "refusal", "list_item",
            "yes_no", "q_n", "q_supported", "q_mismatch", "q_unit_mismatch", "q_absent"]
