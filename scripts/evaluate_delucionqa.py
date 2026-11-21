"""DelucionQA evaluation: train on train, choose every threshold on dev, report test once.

Writes results/delucionqa.json and results/delucionqa.md. Test numbers are reported on all 671 test answers and on the
630 whose contexts were reconstructed without any unknown word.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.baselines import sim_overlap_predict  # noqa: E402
from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.decide import SentenceModel, abstention_curve, answer_scores, tune_threshold  # noqa: E402
from claimgate.features import FEATURES  # noqa: E402
from claimgate.metrics import bootstrap_macro_f1, macro_f1, paired_bootstrap, report  # noqa: E402
from claimgate.pipeline import ClaimGate  # noqa: E402

PUBLISHED = {"Sim-overlap": 71.09, "Sim-hybrid": 70.81, "Sim-cosine": 69.45, "Keyword-match": 52.77}
NLI_COLS = ["e_win_max", "c_at_best", "n_at_best", "c_win_max", "e_win_mean", "e_chunk_max", "c_chunk_max", "n_chunk_min"]
Q_COLS = ["q_n", "q_supported", "q_mismatch", "q_unit_mismatch", "q_absent"]


def unsup_score(df: pd.DataFrame, use_quantities: bool) -> np.ndarray:
    s = 1 - np.maximum(df.e_win_max, df.e_chunk_max).to_numpy()
    if use_quantities:
        s = np.where((df.q_mismatch + df.q_unit_mismatch).to_numpy() > 0, 1.0, s)
    return s


def evaluate(name: str, dev: pd.DataFrame, test: pd.DataFrame, dev_s: np.ndarray, test_s: np.ndarray, complete: set) -> dict:
    da, ta = answer_scores(dev, dev_s), answer_scores(test, test_s)
    t, dev_f = tune_threshold(da.label.to_numpy(), da.score.to_numpy())
    y, p = ta.label.to_numpy(), (ta.score.to_numpy() >= t).astype(int)
    mask = ta.index.isin(complete)
    r = {"system": name, "threshold": t, "dev_macro_f1": dev_f, "test": report(y, p, ta.score.to_numpy()),
         "test_ci95": bootstrap_macro_f1(y, p), "test_complete_contexts": report(y[mask], p[mask]),
         "abstention": abstention_curve(y, ta.score.to_numpy(), t)}
    r["_pred"] = pd.Series(p, index=ta.index)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="deberta-v3-large-nli")
    args = ap.parse_args()
    fdir = ROOT / "artifacts" / "features"
    tr, dv, te = (pd.read_parquet(fdir / f"{s}_{args.tag}.parquet") for s in ("train", "dev", "test"))
    complete = set(te.loc[te.coverage == 1.0, "uid"])
    results = []

    dev_ex, test_ex = load_delucionqa("dev"), load_delucionqa("test")
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t_so = max(grid, key=lambda t: macro_f1([e.label for e in dev_ex], sim_overlap_predict(dev_ex, t)))
    so = pd.Series(sim_overlap_predict(test_ex, t_so), index=[e.uid for e in test_ex])
    y_so = pd.Series([e.label for e in test_ex], index=so.index)
    mask = so.index.isin(complete)
    results.append({"system": "Sim-overlap (Bosch method, our run)", "threshold": t_so, "test": report(y_so, so),
                    "test_ci95": bootstrap_macro_f1(y_so, so), "test_complete_contexts": report(y_so[mask], so[mask]), "_pred": so})

    results.append(evaluate("NLI zero-shot (DeBERTa-v3-large)", dv, te, unsup_score(dv, False), unsup_score(te, False), complete))
    results.append(evaluate("NLI zero-shot + quantity checks", dv, te, unsup_score(dv, True), unsup_score(te, True), complete))
    for name, feats in [("ClaimGate (all evidence features)", FEATURES),
                        ("ClaimGate without NLI", [f for f in FEATURES if f not in NLI_COLS]),
                        ("ClaimGate without quantity checks", [f for f in FEATURES if f not in Q_COLS])]:
        m = SentenceModel.fit(tr, feats)
        results.append(evaluate(name, dv, te, m.predict_proba(dv), m.predict_proba(te), complete))
        if feats is FEATURES:
            ClaimGate(None, m, results[-1]["threshold"]).save()

    ref = results[0]["_pred"]
    for r in results[1:]:
        p = r["_pred"].reindex(ref.index)
        r["vs_sim_overlap"] = paired_bootstrap(y_so.to_numpy(), p.to_numpy(), ref.to_numpy())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "_pred"} for r in results]
    (out / "delucionqa.json").write_text(json.dumps({"published_test_macro_f1": PUBLISHED, "results": clean}, indent=1))
    lines = ["| System | Test macro F1 (671) | 95% CI | Complete contexts (630) | Hallucinated P / R | Dev macro F1 |", "|---|---|---|---|---|---|"]
    for k, v in PUBLISHED.items():
        lines.append(f"| {k} (published, Sadat et al. 2023) | {v:.2f} | | | | |")
    for r in clean:
        t = r["test"]
        ci = r["test_ci95"]
        lines.append(f"| {r['system']} | {100 * t['macro_f1']:.2f} | {100 * ci[0]:.1f} to {100 * ci[1]:.1f} | "
                     f"{100 * r['test_complete_contexts']['macro_f1']:.2f} | {100 * t['hallucinated_precision']:.1f} / {100 * t['hallucinated_recall']:.1f} | "
                     f"{100 * r.get('dev_macro_f1', float('nan')):.2f} |")
    (out / "delucionqa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in clean[1:]:
        print(r["system"], "vs Sim-overlap:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["vs_sim_overlap"].items()})


if __name__ == "__main__":
    main()
