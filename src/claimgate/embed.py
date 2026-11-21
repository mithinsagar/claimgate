"""Sentence-embedding similarity between a claim and the evidence sentences (bge-small-en-v1.5, runs on CPU)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from claimgate.data import ROOT

MODEL_DIR = ROOT / "models" / "bge-small-en-v1.5"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(MODEL_DIR), device="cpu")


def encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return model().encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def max_similarity(claims: list[str], evidence_sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per claim: best cosine similarity to any evidence sentence, and the mean of the best two."""
    c, e = encode(claims), encode(evidence_sentences or [""])
    sims = c @ e.T
    top = np.sort(sims, axis=1)[:, ::-1]
    return top[:, 0], top[:, : min(2, top.shape[1])].mean(axis=1)
