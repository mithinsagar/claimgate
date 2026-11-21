"""Decision layer: sentence-level risk model, answer-level aggregation, calibration and abstention.

The sentence model is a standardised logistic regression over evidence features, trained on DelucionQA train only. It
beat gradient boosting on every feature set in 5-fold grouped cross-validation on train (answer-level AUC 0.874 vs
0.870 with all features), so the simpler model is the default; "gbm" stays available. An answer is hallucinated when its
riskiest sentence crosses a threshold chosen on dev. Probabilities are calibrated with isotonic regression fitted on
out-of-fold train predictions, so the abstention band is expressed in calibrated probability rather than raw scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from claimgate.metrics import macro_f1


def gbm(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                                          random_state=seed)


def logreg(seed: int = 0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, random_state=seed))


@dataclass
class SentenceModel:
    features: list[str]
    models: list
    calibrator: IsotonicRegression

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str], seeds: int = 5, folds: int = 5, kind: str = "lr") -> SentenceModel:
        make = logreg if kind == "lr" else gbm
        seeds = 1 if kind == "lr" else seeds
        X, y, groups = df[features].to_numpy(), df["target"].to_numpy(), df["uid"].to_numpy()
        oof = np.zeros(len(df))
        for tr, va in GroupKFold(folds).split(X, y, groups):
            oof[va] = np.mean([make(s).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for s in range(seeds)], axis=0)
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
        models = [make(s).fit(X, y) for s in range(seeds)]
        return cls(features, models, cal)

    def raw(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy()
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.raw(df))


def answer_scores(df: pd.DataFrame, sent_scores: np.ndarray) -> pd.DataFrame:
    """Answer risk = max sentence risk (an answer is hallucinated if any sentence is unsupported)."""
    d = df[["uid", "answer_label"]].copy()
    d["s"] = sent_scores
    g = d.groupby("uid", sort=False)
    return pd.DataFrame({"label": g["answer_label"].first(), "score": g["s"].max(), "noisy_or": g["s"].apply(lambda v: 1 - np.prod(1 - v))})


def tune_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    grid = np.unique(np.round(scores, 4))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = macro_f1(labels, scores >= t)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def margin(scores, threshold: float) -> np.ndarray:
    """Distance from the decision threshold in logit space (chosen over probability space on dev)."""
    return np.abs(logit(scores) - logit(threshold))


def review_band(scores: np.ndarray, threshold: float, review_share: float = 0.2) -> float:
    """Logit margin below which an answer is sent to review, sized so `review_share` of the given answers are."""
    return float(np.quantile(margin(scores, threshold), review_share))


def abstention_curve(labels: np.ndarray, scores: np.ndarray, threshold: float, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> list[dict]:
    """Accuracy and macro F1 on the answers the system still decides when it abstains on the closest calls."""
    order = np.argsort(-margin(scores, threshold))
    out = []
    for cov in coverages:
        keep = order[: max(2, int(round(cov * len(scores))))]
        pred = scores[keep] >= threshold
        out.append({"coverage": cov, "accuracy": float((pred == labels[keep]).mean()), "macro_f1": macro_f1(labels[keep], pred), "n": int(len(keep))})
    return out


def contradiction_gate(df: pd.DataFrame) -> np.ndarray:
    """Claims whose words are grounded in the evidence but whose meaning is contradicted.

    Some evidence window contradicts the claim (p > 0.9), no window or chunk entails it (p < 0.1), and under 20% of its
    content words are new. DelucionQA train has 17 contradicted sentences out of 4,433, so the learned model gives these
    signals little weight and misses negation or condition flips, which keep the vocabulary intact. The gate is a fixed
    rule; it changes no decision on DelucionQA dev.
    """
    emax = np.maximum(df.e_win_max.to_numpy(), df.e_chunk_max.to_numpy())
    return (df.c_win_max.to_numpy() > 0.9) & (emax < 0.1) & (df.novel_frac.to_numpy() < 0.2)


def gated_risk(model: SentenceModel, df: pd.DataFrame) -> np.ndarray:
    return np.where(contradiction_gate(df), 1.0, model.predict_proba(df))
