"""Run QuantStress (controlled edits of supported DelucionQA dev+test claims) through each system at its dev-chosen threshold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.features import sentence_features  # noqa: E402
from claimgate.nli import NLI  # noqa: E402
from claimgate.pipeline import MODEL_PATH  # noqa: E402
from claimgate.stress import as_example, generate  # noqa: E402

KINDS = ["original", "converted", "value_shift", "unit_swap", "negation", "condition_swap"]


def main() -> None:
    res = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    thr = {r["system"]: r["threshold"] for r in res["results"]}
    import joblib
    gate = joblib.load(MODEL_PATH)
    nli = NLI()
    items = generate(load_delucionqa("dev") + load_delucionqa("test"))
    exs = [as_example(it, ex, i) for i, (it, ex) in enumerate(items)]
    df = sentence_features(exs, nli)
    raw = sentence_features(exs, nli, canonical=False)
    kinds = np.array([it.kind for it, _ in items])
    qbad = (df.q_mismatch + df.q_unit_mismatch).to_numpy() > 0
    systems = {
        "NLI zero-shot, no unit canonicalisation": (1 - np.maximum(raw.e_win_max, raw.e_chunk_max).to_numpy()) >= thr["NLI zero-shot (DeBERTa-v3-large)"],
        "NLI zero-shot": (1 - np.maximum(df.e_win_max, df.e_chunk_max).to_numpy()) >= thr["NLI zero-shot (DeBERTa-v3-large)"],
        "NLI + quantity checks": qbad | ((1 - np.maximum(df.e_win_max, df.e_chunk_max).to_numpy()) >= thr["NLI zero-shot + quantity checks"]),
        "ClaimGate": qbad | (gate["model"].predict_proba(df) >= gate["threshold"]),
    }
    rows = []
    for name, flagged in systems.items():
        row = {"system": name}
        for k in KINDS:
            m = kinds == k
            row[k] = float(flagged[m].mean()) if m.any() else float("nan")
        rows.append(row)
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
