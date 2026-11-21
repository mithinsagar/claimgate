"""Metrics. DelucionQA's headline metric is Macro F1 over answer-level labels (Hallucinated / Not Hallucinated)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro"))


def report(y_true, y_pred, scores=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    out = {"macro_f1": macro_f1(y_true, y_pred), "hallucinated_precision": float(p[0]), "hallucinated_recall": float(r[0]),
           "hallucinated_f1": float(f[0]), "accuracy": float((y_true == y_pred).mean()), "n": int(len(y_true))}
    if scores is not None and 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    return out


def bootstrap_macro_f1(y_true, y_pred, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(macro_f1(y_true[idx], y_pred[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y_true, pred_a, pred_b, n: int = 10000, seed: int = 0) -> dict:
    """Probability that system A's macro F1 exceeds B's under resampling of test answers."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs.append(macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx]))
    diffs = np.asarray(diffs)
    return {"mean_diff": float(diffs.mean()), "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "p_a_not_better": float((diffs <= 0).mean())}


def best_threshold(y_true, scores, grid=None) -> tuple[float, float]:
    """Threshold on a score (higher = more likely hallucinated) maximising macro F1; used on dev only."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199))) if grid is None else grid
    best = max(((macro_f1(y_true, scores >= t), t) for t in grid), key=lambda x: x[0])
    return float(best[1]), float(best[0])


def expected_calibration_error(y_true, prob, bins: int = 10) -> float:
    """ECE with equal-width bins over predicted probability of the positive class."""
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())
    return float(ece)
