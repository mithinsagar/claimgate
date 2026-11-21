"""Out-of-domain check on RAGTruth QA (response level), with models trained only on DelucionQA (car manual).

Thresholds: either the one chosen on DelucionQA dev (no RAGTruth data at all) or one re-picked on 300 random RAGTruth train
responses (threshold only; nothing is retrained). Metric as in the RAGTruth paper: precision / recall / F1 of the
hallucinated class over the 900 QA test responses. The zero-shot detectors get their threshold from the same 300 responses.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT  # noqa: E402
from claimgate.decide import answer_scores, contradiction_gate  # noqa: E402
from claimgate.features import sentence_features  # noqa: E402
from claimgate.nli import NLI  # noqa: E402
from claimgate.pipeline import FULL_MODEL_PATH, MODEL_PATH  # noqa: E402
from claimgate.ragtruth import load_ragtruth_qa  # noqa: E402
from claimgate.signals import load_scores  # noqa: E402

PUBLISHED_QA = {"Prompt gpt-3.5-turbo": (18.8, 84.4, 30.8), "Prompt gpt-4-turbo": (33.2, 90.6, 45.6), "SelfCheckGPT gpt-3.5-turbo": (35.0, 58.0, 43.7),
                "LMvLM gpt-4-turbo": (18.7, 76.9, 30.1), "Fine-tuned Llama-2-13B (trained on RAGTruth)": (61.6, 76.3, 68.2)}


def prf(y, p) -> dict:
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    return {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f[0])}


def best_f1_threshold(y, s) -> float:
    return float(max(np.unique(np.round(s, 4)), key=lambda t: prf(y, s >= t)["f1"]))


def f1_ci(y, p, n=5000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    vals = [prf(y[i], p[i])["f1"] for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    res = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    use_gate = res["design_decisions"].get("use_gate", True)
    fdir = ROOT / "artifacts" / "features"
    frames, nli = {}, None
    for name, loader in (("ragtruth_dev300", lambda: load_ragtruth_qa("train", limit=300)), ("ragtruth_test", lambda: load_ragtruth_qa("test"))):
        path = fdir / f"{name}.parquet"
        if not path.exists():
            nli = nli or NLI(verbose=True)
            sentence_features(loader(), nli).to_parquet(path)
        frames[name] = load_scores(name, pd.read_parquet(path))
    dev, test = frames["ragtruth_dev300"], frames["ragtruth_test"]
    y_test = answer_scores(test, np.zeros(len(test))).label.to_numpy()

    systems = {}

    def zs(d):
        return 1 - np.maximum(d.e_win_max, d.e_chunk_max).to_numpy()
    systems["NLI zero-shot (DeBERTa-v3-large)"] = (zs(dev), zs(test), None)
    for col, name in (("hhem", "HHEM-2.1-Open zero-shot"), ("minicheck", "MiniCheck-Flan-T5-Large zero-shot")):
        if col in test and test[col].notna().all():
            systems[name] = (1 - dev[col].to_numpy(), 1 - test[col].to_numpy(), None)
    for name, path in (("ClaimGate", FULL_MODEL_PATH), ("ClaimGate, no fine-tuned network", MODEL_PATH),
                       ("ClaimGate variant: fine-tuned verifier", ROOT / "models" / "claimgate_verifier.joblib")):
        if not Path(path).exists():
            continue
        g = joblib.load(path)
        if not all(c in test and test[c].notna().all() for c in g["model"].features):
            continue
        for label, quantity_override in (("", True), (" without quantity override", False)):
            def risk(d, g=g, q=quantity_override):
                r = g["model"].predict_proba(d)
                if use_gate:
                    r = np.where(contradiction_gate(d), 1.0, r)
                if q:
                    r = np.where((d.q_mismatch + d.q_unit_mismatch).to_numpy() > 0, 1.0, r)
                return r
            systems[name + label] = (risk(dev), risk(test), g["threshold"])

    results = {"n_test": int(len(y_test)), "hallucinated": int(y_test.sum()), "systems": {}}
    for name, (r_dev, r_test, dqa_thr) in systems.items():
        da, ta = answer_scores(dev, r_dev), answer_scores(test, r_test)
        s = ta.score.to_numpy()
        entry = {"roc_auc": float(roc_auc_score(y_test, s))}
        thresholds = {"RAGTruth 300": best_f1_threshold(da.label.to_numpy(), da.score.to_numpy())}
        if dqa_thr is not None:
            thresholds["DelucionQA dev"] = dqa_thr
        for tname, t in thresholds.items():
            p = (s >= t).astype(int)
            entry[f"threshold from {tname}"] = {"threshold": t, **prf(y_test, p), "f1_ci95": f1_ci(y_test, p)}
        results["systems"][name] = entry
    out = ROOT / "results"
    (out / "ragtruth_qa.json").write_text(json.dumps({"published_qa_response_level": PUBLISHED_QA, **results}, indent=1))
    lines = ["| System | Precision | Recall | F1 | AUC |", "|---|---|---|---|---|"]
    for k, (pr, rc, f) in PUBLISHED_QA.items():
        lines.append(f"| {k} (published, Niu et al. 2024) | {pr:.1f} | {rc:.1f} | {f:.1f} | |")
    for name, entry in results["systems"].items():
        for key in (k for k in entry if k.startswith("threshold")):
            r = entry[key]
            lines.append(f"| {name}, {key} | {100 * r['precision']:.1f} | {100 * r['recall']:.1f} | {100 * r['f1']:.1f} "
                         f"({100 * r['f1_ci95'][0]:.1f} to {100 * r['f1_ci95'][1]:.1f}) | {entry['roc_auc']:.3f} |")
    (out / "ragtruth_qa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
