"""QuantStress: controlled edits of supported DelucionQA dev+test claims, through every system at its dev-chosen threshold.

The edits come from dev and test answers only; the rule-made contradictions used in training come from train answers,
with sentences that also occur verbatim in dev or test excluded, so this set is held out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.decide import contradiction_gate  # noqa: E402
from claimgate.features import sentence_features  # noqa: E402
from claimgate.nli import NLI  # noqa: E402
from claimgate.pipeline import FULL_MODEL_PATH, MODEL_PATH  # noqa: E402
from claimgate.signals import load_scores  # noqa: E402
from claimgate.stress import as_example, generate  # noqa: E402

KINDS = ["original", "converted", "value_shift", "unit_swap", "negation", "condition_swap"]


def main() -> None:
    res = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    thr = {r["system"]: r["threshold"] for r in res["results"]}
    use_gate = res["design_decisions"].get("use_gate", True)
    items = generate(load_delucionqa("dev") + load_delucionqa("test"))
    kinds = np.array([it.kind for it, _ in items])
    fdir = ROOT / "artifacts" / "features"
    cached, raw_path = fdir / "quantstress.parquet", fdir / "quantstress_raw.parquet"
    if cached.exists() and raw_path.exists() and len(pd.read_parquet(cached)) == len(items):
        df, raw = pd.read_parquet(cached), pd.read_parquet(raw_path)
    else:
        nli = NLI(verbose=True)
        exs = [as_example(it, ex, i) for i, (it, ex) in enumerate(items)]
        df = sentence_features(exs, nli).assign(kind=kinds)
        raw = sentence_features(exs, nli, canonical=False)
        df.to_parquet(cached)
        raw.to_parquet(raw_path)
    df = load_scores("quantstress", df)
    qbad = (df.q_mismatch + df.q_unit_mismatch).to_numpy() > 0
    gate = contradiction_gate(df) if use_gate else np.zeros(len(df), bool)

    def zs(d):
        return 1 - np.maximum(d.e_win_max, d.e_chunk_max).to_numpy()
    systems = {"NLI zero-shot, no unit canonicalisation": zs(raw) >= thr["NLI zero-shot (DeBERTa-v3-large)"],
               "NLI zero-shot": zs(df) >= thr["NLI zero-shot (DeBERTa-v3-large)"]}
    for col, name in (("hhem", "HHEM-2.1-Open zero-shot"), ("minicheck", "MiniCheck-Flan-T5-Large zero-shot"), ("ft_sup", "Fine-tuned verifier alone")):
        if col in df and df[col].notna().all() and name in thr:
            systems[name] = (1 - df[col].to_numpy()) >= thr[name]
    for name, path in (("ClaimGate, no fine-tuned network", MODEL_PATH), ("ClaimGate", FULL_MODEL_PATH)):
        if not Path(path).exists():
            continue
        g = joblib.load(path)
        if all(c in df and df[c].notna().all() for c in g["model"].features):
            systems[name] = qbad | gate | (g["model"].predict_proba(df) >= g["threshold"])
    rows = [{"system": name, **{k: float(flagged[kinds == k].mean()) for k in KINDS}} for name, flagged in systems.items()]
    table = pd.DataFrame(rows).set_index("system")
    counts = {k: int((kinds == k).sum()) for k in KINDS}
    out = ROOT / "results"
    (out / "quantstress.json").write_text(json.dumps({"counts": counts, "flag_rate": table.to_dict(orient="index")}, indent=1))
    hdr = "| System | " + " | ".join(f"{k} (n={counts[k]})" for k in KINDS) + " |"
    lines = ["Share of claims flagged. Lower is better for original and converted; higher is better for the four edits.", "", hdr,
             "|" + "---|" * (len(KINDS) + 1)]
    for name, r in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{100 * r[k]:.1f}%" for k in KINDS) + " |")
    (out / "quantstress.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
