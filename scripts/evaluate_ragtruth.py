"""Out-of-domain check on RAGTruth QA (response level), with the model trained only on DelucionQA (car manual).

Two thresholds are reported: the one chosen on DelucionQA dev (no RAGTruth data used at all) and one chosen on 300 random
RAGTruth train responses (threshold only, the model is not retrained). The metric follows the RAGTruth paper:
precision / recall / F1 of the hallucinated class over the 900 QA test responses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT  # noqa: E402
from claimgate.decide import answer_scores, gated_risk  # noqa: E402
from claimgate.features import sentence_features  # noqa: E402
from claimgate.metrics import bootstrap_macro_f1  # noqa: E402
from claimgate.nli import NLI  # noqa: E402
from claimgate.pipeline import MODEL_PATH  # noqa: E402
from claimgate.ragtruth import load_ragtruth_qa  # noqa: E402

PUBLISHED_QA = {"Prompt gpt-3.5-turbo": (18.8, 84.4, 30.8), "Prompt gpt-4-turbo": (33.2, 90.6, 45.6), "SelfCheckGPT gpt-3.5-turbo": (35.0, 58.0, 43.7),
                "LMvLM gpt-4-turbo": (18.7, 76.9, 30.1), "Fine-tuned Llama-2-13B (trained on RAGTruth)": (61.6, 76.3, 68.2)}


def prf(y, p) -> dict:
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    return {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f[0])}


def best_f1_threshold(y, s) -> float:
    return float(max(np.unique(np.round(s, 4)), key=lambda t: prf(y, s >= t)["f1"]))


def main() -> None:
    gate = joblib.load(MODEL_PATH)
    nli = NLI(verbose=True)
    fdir = ROOT / "artifacts" / "features"
    frames = {}
    for name, exs in (("ragtruth_dev300", load_ragtruth_qa("train", limit=300)), ("ragtruth_test", load_ragtruth_qa("test"))):
        path = fdir / f"{name}.parquet"
        if not path.exists():
            sentence_features(exs, nli).to_parquet(path)
        frames[name] = __import__("pandas").read_parquet(path)
    dev, test = frames["ragtruth_dev300"], frames["ragtruth_test"]
    da = answer_scores(dev, gated_risk(gate["model"], dev))
    ta = answer_scores(test, gated_risk(gate["model"], test))
    y = ta.label.to_numpy()
    s = ta.score.to_numpy()
    from sklearn.metrics import roc_auc_score
    results = {"n_test": int(len(y)), "hallucinated": int(y.sum()), "roc_auc": float(roc_auc_score(y, s))}
    for name, t in (("threshold from DelucionQA dev", gate["threshold"]),
                    ("threshold from 300 RAGTruth train responses", best_f1_threshold(da.label.to_numpy(), da.score.to_numpy()))):
        p = (s >= t).astype(int)
        lo, hi = bootstrap_macro_f1(y, p)
        rng = np.random.default_rng(0)
        f1s = []
        for _ in range(5000):
            idx = rng.integers(0, len(y), len(y))
            f1s.append(prf(y[idx], p[idx])["f1"])
        results[name] = {"threshold": t, **prf(y, p), "f1_ci95": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
                         "macro_f1_ci95": (lo, hi)}
    out = ROOT / "results"
    (out / "ragtruth_qa.json").write_text(json.dumps({"published_qa_response_level": PUBLISHED_QA, "claimgate": results}, indent=1))
    lines = ["| System | Precision | Recall | F1 |", "|---|---|---|---|"]
    for k, (pr, rc, f) in PUBLISHED_QA.items():
        lines.append(f"| {k} (published, Niu et al. 2024) | {pr:.1f} | {rc:.1f} | {f:.1f} |")
    for k in ("threshold from DelucionQA dev", "threshold from 300 RAGTruth train responses"):
        r = results[k]
        lines.append(f"| ClaimGate trained on DelucionQA only, {k} | {100 * r['precision']:.1f} | {100 * r['recall']:.1f} | {100 * r['f1']:.1f} "
                     f"({100 * r['f1_ci95'][0]:.1f} to {100 * r['f1_ci95'][1]:.1f}) |")
    (out / "ragtruth_qa.md").write_text("\n".join(lines) + f"\n\nROC AUC {results['roc_auc']:.3f} on {len(y)} test responses.\n")
    print("\n".join(lines))
    print("ROC AUC", round(results["roc_auc"], 4))


if __name__ == "__main__":
    main()
