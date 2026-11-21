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


def abstention_curve(labels: np.ndarray, scores: np.ndarray, threshold: float, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)) -> list[dict]:
    """Macro F1 when the system abstains on the answers whose calibrated risk is closest to the threshold."""
    margin = np.abs(scores - threshold)
    order = np.argsort(-margin)
    out = []
    for cov in coverages:
        keep = order[: max(2, int(round(cov * len(scores))))]
        out.append({"coverage": cov, "macro_f1": macro_f1(labels[keep], scores[keep] >= threshold), "n": int(len(keep))})
    return out
