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
from claimgate.metrics import (  # noqa: E402
    bootstrap_macro_f1,
    expected_calibration_error,  # noqa: E402
    macro_f1,
    paired_bootstrap,
    report,
)
from claimgate.pipeline import LLM_MODEL_PATH, MODEL_PATH, ClaimGate  # noqa: E402

PUBLISHED = {"Sim-overlap": 71.09, "Sim-hybrid": 70.81, "Sim-cosine": 69.45, "Keyword-match": 52.77}
NLI_COLS = ["e_win_max", "c_at_best", "n_at_best", "c_win_max", "e_win_mean", "e_chunk_max", "c_chunk_max", "n_chunk_min"]
Q_COLS = ["q_n", "q_supported", "q_mismatch", "q_unit_mismatch", "q_absent"]


def with_llm(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach the LLM judge's per-sentence P(supported) when artifacts/llm/<split>.jsonl covers every answer."""
    path = ROOT / "artifacts" / "llm" / f"{split}.jsonl"
    if not path.exists():
        return df
    rows = [(r["uid"], j, p) for r in map(json.loads, open(path)) for j, p in enumerate(r["p_yes"])]
    llm = pd.DataFrame(rows, columns=["uid", "sent_idx", "llm_yes"]).drop_duplicates(["uid", "sent_idx"])
    out = df.merge(llm, on=["uid", "sent_idx"], how="left")
    return out if out.llm_yes.notna().all() else df


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
    tr, dv, te = (with_llm(pd.read_parquet(fdir / f"{s}_{args.tag}.parquet"), s) for s in ("train", "dev", "test"))
    has_llm = all("llm_yes" in d for d in (tr, dv, te))
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
    if has_llm:
        results.append(evaluate("LLM judge zero-shot (Qwen2.5-7B-Instruct, 4-bit)", dv, te, 1 - dv.llm_yes.to_numpy(), 1 - te.llm_yes.to_numpy(), complete))
    systems = [("ClaimGate light (no LLM)", FEATURES, MODEL_PATH),
               ("ClaimGate light without NLI", [f for f in FEATURES if f not in NLI_COLS], None),
               ("ClaimGate light without quantity checks", [f for f in FEATURES if f not in Q_COLS], None)]
    if has_llm:
        systems.insert(0, ("ClaimGate (evidence features + LLM judge)", FEATURES + ["llm_yes"], LLM_MODEL_PATH))
    for name, feats, path in systems:
        m = SentenceModel.fit(tr, feats)
        r = evaluate(name, dv, te, m.predict_proba(dv), m.predict_proba(te), complete)
        r["sentence_ece_test"] = expected_calibration_error(te.target, m.predict_proba(te))
        results.append(r)
        if path:
            ClaimGate(None, m, r["threshold"]).save(path)

    ref = results[0]["_pred"]
    for r in results[1:]:
        p = r["_pred"].reindex(ref.index)
        r["vs_sim_overlap"] = paired_bootstrap(y_so.to_numpy(), p.to_numpy(), ref.to_numpy())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "_pred"} for r in results]
    (out / "delucionqa.json").write_text(json.dumps({"published_test_macro_f1": PUBLISHED, "results": clean}, indent=1))
    lines = ["| System | Test macro F1 (671) | 95% CI | Complete contexts (630) | Hallucinated P / R | Test AUC | Dev macro F1 |",
             "|---|---|---|---|---|---|---|"]
    for k, v in PUBLISHED.items():
        lines.append(f"| {k} (published, Sadat et al. 2023) | {v:.2f} | | | | | |")
    for r in clean:
        t, ci = r["test"], r["test_ci95"]
        auc = f"{t['roc_auc']:.3f}" if "roc_auc" in t else ""
        dev = f"{100 * r['dev_macro_f1']:.2f}" if "dev_macro_f1" in r else ""
        lines.append(f"| {r['system']} | {100 * t['macro_f1']:.2f} | {100 * ci[0]:.1f} to {100 * ci[1]:.1f} | "
                     f"{100 * r['test_complete_contexts']['macro_f1']:.2f} | {100 * t['hallucinated_precision']:.1f} / "
                     f"{100 * t['hallucinated_recall']:.1f} | {auc} | {dev} |")
    (out / "delucionqa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in clean[1:]:
        print(r["system"], "vs Sim-overlap:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["vs_sim_overlap"].items()})


if __name__ == "__main__":
    main()
