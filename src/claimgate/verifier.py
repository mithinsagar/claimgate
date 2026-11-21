"""Fine-tuned claim verifier: DeBERTa-v3-large NLI adapted to "is this answer sentence supported by the evidence".

Trained by scripts/train_verifier.py with 3-fold cross-fitting on DelucionQA train (plus rule-made contradictions of
train sentences), so every train sentence also gets an out-of-fold score for the decision layer. At inference the three
fold models are averaged. The premise is the claim's most relevant evidence windows in document order, which keeps
inputs short enough to train on a 16 GB machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimgate.data import ROOT
from claimgate.text import content_words, stem

VERIFIER_DIR = ROOT / "models" / "verifier"
MAX_LEN = 320


def premise(ctx_sents: list[str], claim: str, k: int = 4, max_words: int = 260) -> str:
    """The claim's k best 2-sentence windows by content-word overlap, merged and kept in document order."""
    if not ctx_sents:
        return ""
    cw = {stem(w) for w in content_words(claim)}
    wins = [(i, i + 2) for i in range(max(1, len(ctx_sents) - 1))]
    score = [len(cw & {stem(w) for s in ctx_sents[a:b] for w in content_words(s)}) for a, b in wins]
    keep = set()
    for i in sorted(range(len(wins)), key=lambda i: -score[i])[:k]:
        keep.update(range(*wins[i]))
    out, n = [], 0
    for i in sorted(keep):
        if i >= len(ctx_sents):
            continue
        w = len(ctx_sents[i].split())
        if out and n + w > max_words:
            break
        out.append(ctx_sents[i])
        n += w
    return " ".join(out)


class Verifier:
    """Average of the cross-fitted fold models; returns P(supported), P(neither), P(contradicted) per pair."""

    def __init__(self, root: Path = VERIFIER_DIR, device: str | None = None, batch_size: int = 16, dirs: list[Path] | None = None):
        from claimgate.nli import device as default_device
        self.dirs = dirs or sorted(p for p in Path(root).glob("fold*") if (p / "config.json").exists())
        if not self.dirs:
            raise FileNotFoundError(f"no verifier folds in {root}")
        self.device = device or default_device()
        self.tok = AutoTokenizer.from_pretrained(self.dirs[0])
        self.models = [AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float16).eval().to(self.device) for d in self.dirs]
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first", max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            probs = np.mean([torch.softmax(m(**enc).logits.float(), -1).cpu().numpy() for m in self.models], axis=0)
            out[idx] = probs
        return out
