"""Benchmark loaders.

DelucionQA (Bosch Research, EMNLP 2023 Findings): answers from retrieval-augmented ChatGPT on the Jeep Gladiator 2023
manual, with human labels per answer sentence (Supported / Neither / Conflicted) and per answer (Hallucinated if any
sentence is not Supported). Contexts are reconstructed locally by scripts/reconstruct_delucionqa.py (lower-cased, Bosch
tokenisation; unknown words appear as "[?]" and are counted in `coverage`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DQA_DIR = ROOT / "data" / "delucionqa"


@dataclass
class Example:
    uid: str
    split: str
    question: str
    context: str                      # lower-cased, Bosch-tokenised context (unknown words as "[?]")
    context_cased: str | None         # original cased passage when it could be matched exactly
    answer: str
    sentences: list[str]
    sentence_labels: list[str]        # Supported / Neither / Conflicted
    label: int                        # 1 = Hallucinated, 0 = Not Hallucinated
    coverage: float = 1.0
    retrieval: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def sentence_targets(self) -> list[int]:
        return [0 if lab == "Supported" else 1 for lab in self.sentence_labels]


def load_delucionqa(split: str) -> list[Example]:
    out = []
    for line in open(DQA_DIR / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r["answer_sentences"]:
            continue
        out.append(Example(uid=r["sample_id"], split=split, question=r["question"], context=r["context_lower"], context_cased=r.get("context"),
                           answer=r["answer"], sentences=r["answer_sentences"], sentence_labels=r["sentence_labels"],
                           label=int(r["label"] == "Hallucinated"), coverage=r.get("coverage", 1.0), retrieval=r.get("retrieval", "")))
    return out


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


def split_context(ex: Example) -> list[str]:
    """Context sentences: the cased passage when available (better for NLI), otherwise the tokenised one split on ' . '."""
    if ex.context_cased:
        parts = [s.strip() for s in _SENT_END.split(ex.context_cased) if s.strip()]
        if parts:
            return parts
    parts = re.split(r"(?<= [.!?]) ", ex.context)
    return [p.strip() for p in parts if p.strip()]


def context_text(ex: Example) -> str:
    return ex.context_cased or ex.context
