"""RAGTruth (Niu et al., ACL 2024) question-answering subset as ClaimGate examples.

A response counts as hallucinated if it has any annotated span, as in the paper's response-level evaluation (spans
marked implicit_true are included, as the paper does). A response sentence is a positive if it overlaps an annotated span.
"""
from __future__ import annotations

import json
import random
import re

from claimgate.baselines import bosch_tokens
from claimgate.data import ROOT, Example

RT = ROOT / "data" / "raw" / "RAGTruth" / "dataset"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+")


def _sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    out, pos = [], 0
    for part in _SPLIT.split(text):
        start = text.find(part, pos)
        end = start + len(part)
        pos = end
        if len(part.strip()) > 2:
            out.append((part.strip(), start, end))
    return out


def _evidence(source_info: dict) -> str:
    passages = re.split(r"\n*passage \d+:", source_info["passages"])
    return "\n".join(p.strip() for p in passages if p.strip())


def load_ragtruth_qa(split: str, limit: int | None = None, seed: int = 0, exclude: set[str] | None = None) -> list[Example]:
    sources = {}
    for line in open(RT / "source_info.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["task_type"] == "QA":
            sources[s["source_id"]] = s
    rows = [r for r in (json.loads(line) for line in open(RT / "response.jsonl", encoding="utf-8"))
            if r["source_id"] in sources and r["split"] == split and f"rt{r['id']}" not in (exclude or set())]
    if limit:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    out = []
    for r in rows:
        src = sources[r["source_id"]]
        evidence = _evidence(src["source_info"])
        sents = _sentences_with_offsets(r["response"])
        if not sents:
            continue
        spans = [(lab["start"], lab["end"]) for lab in r["labels"]]
        labels = ["Neither" if any(s < e2 and s2 < e for s2, e2 in spans) else "Supported" for _, s, e in sents]
        out.append(Example(uid=f"rt{r['id']}", split=f"ragtruth_{split}", question=src["source_info"]["question"],
                           context=" ".join(bosch_tokens(evidence)), context_cased=evidence, answer=r["response"],
                           sentences=[s for s, _, _ in sents], sentence_labels=labels, label=int(bool(r["labels"])),
                           meta={"evidence": evidence, "model": r["model"]}))
    return out


def adaptation_split(n: int | None = 600) -> tuple[list[Example], list[Example]]:
    """RAGTruth QA train responses for domain adaptation: (training responses, the 300 used to pick thresholds).

    The threshold set is the same 300 responses used everywhere else; the training responses are drawn from the rest of
    the train split, so the two never overlap and the 900 test responses are never touched.
    """
    dev = load_ragtruth_qa("train", limit=300)
    train = load_ragtruth_qa("train", limit=n, seed=1, exclude={e.uid for e in dev})  # n=None: every other train response
    return train, dev
