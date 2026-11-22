"""DelucionQA evaluation. Models and design choices on train (grouped cross-validation), thresholds on dev, test once.

Decision rules:
  * signals: a zero-shot signal (HHEM, MiniCheck) joins the feature model if it raises out-of-fold answer AUC on train by
    at least 0.002;
  * rule-made contradictions of train sentences are added to the feature model's training rows unless they cost more than
    0.005 out-of-fold answer AUC on the real train answers;
  * the fine-tuned verifier: its fold models are averaged at inference, so pooled out-of-fold scores (one scale per fold)
    cannot be stacked reliably on train. The final model is chosen by dev macro F1 among three fixed candidates: the
    feature model, the verifier alone, and an equal-weight blend of the two calibrated risks (verifier calibrated on dev);
  * the contradiction gate stays if it catches more of the held-out-fold rule-made contradictions and costs at most
    0.5 points of dev macro F1; the learned contradiction detector replaces the hand rule under the same test, with its
    false-alarm target (3%, 2%, 1% or 0.5% of supported train sentences) picked as the one catching the most edits.
A first full run stacked the pooled out-of-fold verifier scores; train analysis showed per-fold scale differences, so
the verifier is now combined by calibration and blending instead (README, "How the final model was chosen").
Writes results/delucionqa.{json,md}, the shipped zero-shot model (src/claimgate/assets) and the full model (models/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.isotonic import IsotonicRegression  # noqa: E402

from claimgate.baselines import sim_overlap_predict  # noqa: E402
from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.decide import (  # noqa: E402
    BlendModel,
    ContradictionModel,
    SentenceModel,
    abstention_curve,
    answer_scores,
    contradiction_gate,
    oof_risk,
    review_band,
    tune_threshold,
)
from claimgate.features import FEATURES  # noqa: E402
from claimgate.metrics import bootstrap_macro_f1, expected_calibration_error, macro_f1, paired_bootstrap, report  # noqa: E402
from claimgate.pipeline import FULL_MODEL_PATH, MODEL_PATH, ClaimGate  # noqa: E402
from claimgate.signals import load_scores  # noqa: E402

PUBLISHED = {"Sim-overlap": 71.09, "Sim-hybrid": 70.81, "Sim-cosine": 69.45, "Keyword-match": 52.77}
NLI_COLS = ["e_win_max", "c_at_best", "n_at_best", "c_win_max", "e_win_mean", "e_chunk_max", "c_chunk_max", "n_chunk_min", "e_prev", "e_next"]
Q_COLS = ["q_n", "q_supported", "q_mismatch", "q_unit_mismatch", "q_absent"]
TAG = "deberta-v3-large-nli"
VERIFIER_MODEL_PATH = ROOT / "models" / "claimgate_verifier.joblib"


def with_llm(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Attach the LLM judge's per-sentence P(supported) when artifacts/llm/<split>.jsonl covers every answer."""
    path = ROOT / "artifacts" / "llm" / f"{split}.jsonl"
    if not path.exists():
        return df
    rows = [(r["uid"], j, p) for r in map(json.loads, open(path)) for j, p in enumerate(r["p_yes"])]
    llm = pd.DataFrame(rows, columns=["uid", "sent_idx", "llm_yes"]).drop_duplicates(["uid", "sent_idx"])
    out = df.merge(llm, on=["uid", "sent_idx"], how="left")
    return out if out.llm_yes.notna().all() else df


def load(split: str) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "artifacts" / "features" / f"{split}_{TAG}.parquet")
    return with_llm(load_scores(split, df), split)


def answer_auc(df: pd.DataFrame, risk: np.ndarray) -> float:
    a = answer_scores(df, risk)
    return float(roc_auc_score(a.label, a.score))


def dev_f1(dev: pd.DataFrame, risk: np.ndarray) -> float:
    a = answer_scores(dev, risk)
    return tune_threshold(a.label.to_numpy(), a.score.to_numpy())[1]


def evaluate(name, dev, test, dev_s, test_s, complete, unseen) -> dict:
    da, ta = answer_scores(dev, dev_s), answer_scores(test, test_s)
    t, dev_f = tune_threshold(da.label.to_numpy(), da.score.to_numpy())
    y, s = ta.label.to_numpy(), ta.score.to_numpy()
    p = (s >= t).astype(int)
    m_c, m_u = ta.index.isin(complete), ta.index.isin(unseen)
    return {"system": name, "threshold": t, "dev_macro_f1": dev_f, "test": report(y, p, s), "test_ci95": bootstrap_macro_f1(y, p),
            "test_complete_contexts": report(y[m_c], p[m_c]), "test_unseen_contexts": report(y[m_u], p[m_u]),
            "abstention": abstention_curve(y, s, t), "_pred": pd.Series(p, index=ta.index)}


def main() -> None:
    tr, aug, dv, te = load("train"), load("train_aug"), load("dev"), load("test")
    aug = aug.assign(target=1, answer_label=1)
    complete = set(te.loc[te.coverage == 1.0, "uid"])
    train_ctx = {e.context for e in load_delucionqa("train")}
    unseen = {e.uid for e in load_delucionqa("test") if e.context not in train_ctx}
    signals = [c for c in ("hhem", "minicheck", "ft_sup") if all(c in d and d[c].notna().all() for d in (tr, aug, dv, te))]
    print("signals available:", signals)

    # ---- model selection on train only ----
    sel: dict = {}
    real_groups = tr.uid.to_numpy()
    chosen = list(FEATURES)
    sel["zero_shot_features_auc"] = current = answer_auc(tr, oof_risk(tr, chosen, real_groups))
    for sig in [s for s in signals if s != "ft_sup"]:
        auc = answer_auc(tr, oof_risk(tr, chosen + [sig], real_groups))
        sel[f"+{sig}_auc"] = auc
        if auc >= current + 0.002:
            chosen, current = chosen + [sig], auc
    zero_shot_feats, full_feats = list(chosen), list(chosen)
    if "ft_sup" in signals:
        auc = answer_auc(tr, oof_risk(tr, chosen + ["ft_sup", "ft_con"], real_groups))
        sel["+fine_tuned_verifier_auc"] = auc
        if auc >= current + 0.002:
            full_feats = chosen + ["ft_sup", "ft_con"]
    both = pd.concat([tr, aug], ignore_index=True)
    both_groups = np.concatenate([real_groups, aug.source_uid.to_numpy()])
    real = np.r_[np.ones(len(tr), bool), np.zeros(len(aug), bool)]
    r_both = oof_risk(both, full_feats, both_groups)
    sel["real_rows_only_auc"] = answer_auc(tr, oof_risk(tr, full_feats, real_groups))
    sel["with_rule_made_contradictions_auc"] = answer_auc(tr, r_both[real])
    use_aug = sel["with_rule_made_contradictions_auc"] >= sel["real_rows_only_auc"] - 0.005
    sel["use_rule_made_contradictions"] = use_aug
    # catch rate on held-out-fold rule-made rows by the configuration actually used
    if use_aug:
        r_real, r_aug = r_both[real], r_both[~real]
    else:
        r_real, r_aug = oof_risk(tr, full_feats, real_groups), np.zeros(len(aug))
        fold_of = {u: i % 5 for i, u in enumerate(sorted(set(real_groups)))}
        for k in range(5):
            train_mask = np.array([fold_of[u] != k for u in real_groups])
            aug_mask = np.array([fold_of.get(u, -1) == k for u in aug.source_uid])
            if aug_mask.any():
                m_k = SentenceModel.fit(tr[train_mask], full_feats)
                r_aug[aug_mask] = m_k.predict_proba(aug[aug_mask])
    a_cv = answer_scores(tr, r_real)
    t_cv, _ = tune_threshold(a_cv.label.to_numpy(), a_cv.score.to_numpy())
    sel["rule_made_caught_without_gate"] = float((r_aug >= t_cv).mean())
    sel["rule_made_caught_with_gate"] = float(((r_aug >= t_cv) | contradiction_gate(aug)).mean())

    fit_df, fit_groups, fit_mask = (both, both_groups, real) if use_aug else (tr, real_groups, None)

    def fit(feats):
        return SentenceModel.fit(fit_df, feats, groups=fit_groups, calibrate_on=fit_mask)

    # learned contradiction detector vs the hand-written gate (train-derived edits and dev only)
    supported = tr[tr.target == 0]
    contras = {fpr: ContradictionModel.fit(supported, aug, supported.uid.to_numpy(), aug.source_uid.to_numpy(), target_fpr=fpr)
               for fpr in (0.03, 0.02, 0.01, 0.005)}
    gate_model = None

    def risk(m, d, gate):
        r = m.predict_proba(d)
        return np.where(contradiction_gate(d, gate_model), 1.0, r) if gate else r

    full = fit(full_feats)
    sel["dev_macro_f1_with_gate"] = dev_f1(dv, risk(full, dv, True))
    sel["dev_macro_f1_without_gate"] = dev_f1(dv, risk(full, dv, False))
    use_gate = sel["rule_made_caught_with_gate"] > sel["rule_made_caught_without_gate"] and \
        sel["dev_macro_f1_with_gate"] >= sel["dev_macro_f1_without_gate"] - 0.005
    # replace the hand rule by the learned detector: among false-alarm targets of 3%, 2%, 1% and 0.5% on supported train
    # sentences, take the one that, combined with the risk model, catches the most held-out-fold edits while costing at
    # most 0.5 points of dev macro F1 against the hand rule (train-derived edits and dev only; test untouched)
    learned = {}
    for fpr, c in contras.items():
        gate_model = c
        learned[fpr] = {"rule_made_caught": float(((r_aug >= t_cv) | c.oof_edit_flags).mean()), "dev_macro_f1": dev_f1(dv, risk(full, dv, True))}
    gate_model = None
    sel["learned_gate_by_target_fpr"] = learned
    ok = [f for f, v in learned.items() if v["rule_made_caught"] > sel["rule_made_caught_with_gate"]
          and v["dev_macro_f1"] >= sel["dev_macro_f1_with_gate"] - 0.005]
    use_learned = bool(ok)
    sel["use_learned_gate"] = use_learned
    if use_learned:
        best = max(ok, key=lambda f: learned[f]["rule_made_caught"])
        sel["learned_gate_target_fpr"] = best
        sel["learned_gate_features"] = contras[best].features
        gate_model, use_gate = contras[best], True
    sel.update({"use_gate": use_gate, "zero_shot_model_features": zero_shot_feats, "full_model_features": full_feats})
    print("design decisions:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sel.items() if not isinstance(v, list)})

    # ---- test, once per system ----
    results = []
    dev_ex, test_ex = load_delucionqa("dev"), load_delucionqa("test")
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    t_so = max(grid, key=lambda t: macro_f1([e.label for e in dev_ex], sim_overlap_predict(dev_ex, t)))
    so = pd.Series(sim_overlap_predict(test_ex, t_so), index=[e.uid for e in test_ex])
    y_all = pd.Series([e.label for e in test_ex], index=so.index)
    m_c, m_u = so.index.isin(complete), so.index.isin(unseen)
    results.append({"system": "Sim-overlap (Bosch method, our run)", "threshold": t_so, "test": report(y_all, so), "test_ci95": bootstrap_macro_f1(y_all, so),
                    "test_complete_contexts": report(y_all[m_c], so[m_c]), "test_unseen_contexts": report(y_all[m_u], so[m_u]), "_pred": so})

    def zs(d):
        return 1 - np.maximum(d.e_win_max, d.e_chunk_max).to_numpy()
    results.append(evaluate("NLI zero-shot (DeBERTa-v3-large)", dv, te, zs(dv), zs(te), complete, unseen))
    for col, name in (("hhem", "HHEM-2.1-Open zero-shot"), ("minicheck", "MiniCheck-Flan-T5-Large zero-shot"),
                      ("llm_yes", "LLM judge zero-shot (Qwen2.5-7B-Instruct, 4-bit)"), ("ft_sup", "Fine-tuned verifier alone")):
        if col in dv and col in te and dv[col].notna().all() and te[col].notna().all():
            results.append(evaluate(name, dv, te, 1 - dv[col].to_numpy(), 1 - te[col].to_numpy(), complete, unseen))

    final, final_name = full, "feature model"
    if "ft_sup" in signals:
        dv_sent_risk = 1 - dv.ft_sup.to_numpy()
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(dv_sent_risk, dv.target.to_numpy())
        candidates = {"feature model": full, "verifier alone": BlendModel(full, cal, 0.0), "equal-weight blend": BlendModel(full, cal, 0.5)}
        dev_scores = {k: dev_f1(dv, risk(m, dv, use_gate)) for k, m in candidates.items()}
        sel["final_candidates_dev_macro_f1"] = dev_scores
        final_name = max(dev_scores, key=dev_scores.get)
        final = candidates[final_name]
    sel["final_model"] = final_name
    print("final model chosen on dev:", final_name, sel.get("final_candidates_dev_macro_f1"))
    for name, feats, path in (("ClaimGate", None, FULL_MODEL_PATH), ("ClaimGate, no fine-tuned network", zero_shot_feats, MODEL_PATH)):
        m = final if feats is None else fit(feats)
        feats = m.features
        r = evaluate(name, dv, te, risk(m, dv, use_gate), risk(m, te, use_gate), complete, unseen)
        r["sentence_ece_test"] = expected_calibration_error(te.target, m.predict_proba(te))
        r["review_band_logit"] = review_band(answer_scores(dv, risk(m, dv, use_gate)).score.to_numpy(), r["threshold"])
        r["features"] = feats
        results.append(r)
        ClaimGate(None, m, r["threshold"], r["review_band_logit"], contra=gate_model).save(path)
    if "ft_sup" in signals:
        for cname, cm in candidates.items():
            if cname == final_name:
                continue
            r = evaluate(f"ClaimGate variant: {cname} (not selected on dev)", dv, te, risk(cm, dv, use_gate), risk(cm, te, use_gate), complete, unseen)
            r["review_band_logit"] = review_band(answer_scores(dv, risk(cm, dv, use_gate)).score.to_numpy(), r["threshold"])
            results.append(r)
            if cname == "verifier alone":
                ClaimGate(None, cm, r["threshold"], r["review_band_logit"], contra=gate_model).save(VERIFIER_MODEL_PATH)
    for name, feats, gate in (("without contradiction gate", full_feats, not use_gate),
                              ("without NLI features", [f for f in full_feats if f not in NLI_COLS], use_gate),
                              ("without quantity features", [f for f in full_feats if f not in Q_COLS], use_gate),
                              ("without rule-made contradictions in training", full_feats, use_gate)):
        m = SentenceModel.fit(tr, feats) if name.endswith("training") else fit(feats)
        r = evaluate(f"ClaimGate {name}", dv, te, risk(m, dv, gate), risk(m, te, gate), complete, unseen)
        r["train_cv_answer_auc"] = answer_auc(tr, oof_risk(tr, feats, real_groups))
        results.append(r)
    if "llm_yes" in tr:
        sel["+llm_judge_feature_auc"] = answer_auc(tr, oof_risk(tr, full_feats + ["llm_yes"], real_groups))

    head = next(r for r in results if r["system"] == "ClaimGate")
    idx = head["_pred"].index
    y = y_all.reindex(idx).to_numpy()
    mu = idx.isin(unseen)
    for r in results:
        if r is head:
            continue
        other = r["_pred"].reindex(idx).to_numpy()
        r["claimgate_vs_this"] = paired_bootstrap(y, head["_pred"].to_numpy(), other)
        r["claimgate_vs_this_unseen_contexts"] = paired_bootstrap(y[mu], head["_pred"].to_numpy()[mu], other[mu])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    meta = {"n_test": len(y_all), "n_unseen_contexts": len(unseen), "n_complete_contexts": len(complete)}
    (out / "delucionqa.json").write_text(json.dumps({"published_test_macro_f1": PUBLISHED, "design_decisions": sel, **meta, "results": clean},
                                                     indent=1, default=float))
    lines = ["| System | Test macro F1 | 95% CI | Unseen contexts | Hallucinated P / R | Test AUC | p (ClaimGate not better) |",
             "|---|---|---|---|---|---|---|"]
    for k, v in PUBLISHED.items():
        lines.append(f"| {k} (published, Sadat et al. 2023) | {v:.2f} | | | | | |")
    for r in clean:
        t, ci = r["test"], r["test_ci95"]
        auc = f"{t['roc_auc']:.3f}" if "roc_auc" in t else ""
        pv = f"{r['claimgate_vs_this']['p_a_not_better']:.3f}" if "claimgate_vs_this" in r else ""
        unseen_f1 = 100 * r["test_unseen_contexts"]["macro_f1"]
        lines.append(f"| {r['system']} | {100 * t['macro_f1']:.2f} | {100 * ci[0]:.1f} to {100 * ci[1]:.1f} | {unseen_f1:.2f} | "
                     f"{100 * t['hallucinated_precision']:.1f} / {100 * t['hallucinated_recall']:.1f} | {auc} | {pv} |")
    (out / "delucionqa.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
