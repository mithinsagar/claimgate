"""Figures for the README from results/*.json (light surface, fixed palette slots)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimgate.data import ROOT  # noqa: E402

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#b9b8b2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
OUT = ROOT / "docs" / "figures"


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def headline(d: dict) -> dict:
    names = [r["system"] for r in d["results"]]
    return d["results"][names.index("ClaimGate")]


def delucionqa():
    d = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    top = headline(d)["system"]
    rows = [(f"{k} (paper)", v, None, False) for k, v in d["published_test_macro_f1"].items()]
    for r in d["results"]:
        if r["system"].startswith("ClaimGate without") or r["system"].startswith("ClaimGate with "):
            continue  # ablations live in the table
        ci = r.get("test_ci95")
        rows.append((r["system"], 100 * r["test"]["macro_f1"], (100 * ci[0], 100 * ci[1]) if ci else None, r["system"] == top))
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.1), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    for i, (name, v, ci, main) in enumerate(rows):
        color = SERIES[0] if main else (MUTED if "(paper)" in name else "#8fb5e6")
        ax.barh(i, v, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        if ci:
            ax.plot(ci, [i, i], color=INK2, linewidth=1.2)
        ax.text(v + 0.6 if not ci else ci[1] + 0.6, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK, fontweight="bold" if main else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(40, 100)
    ax.set_xlabel("DelucionQA test macro F1 (whiskers: 95% bootstrap CI)", color=INK2, fontsize=9)
    ax.axvline(71.09, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(71.4, len(rows) - 0.45, "best published: 71.09", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "delucionqa.png", facecolor=SURFACE)


def quantstress():
    p = ROOT / "results" / "quantstress.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    kinds = ["original", "converted", "value_shift", "unit_swap", "negation", "condition_swap"]
    labels = ["original\n(should pass)", "unit converted\n(should pass)", "value shifted", "unit swapped", "negation flipped", "condition swapped"]
    colors = {"NLI zero-shot": SERIES[1], "HHEM-2.1-Open zero-shot": SERIES[2], "MiniCheck-Flan-T5-Large zero-shot": SERIES[3],
              "ClaimGate, no fine-tuned network": SERIES[4], "ClaimGate": SERIES[0]}
    systems = [s for s in colors if s in d["flag_rate"]]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    w = 0.8 / len(systems)
    for j, s in enumerate(systems):
        vals = [100 * d["flag_rate"][s][k] for k in kinds]
        xs = [i + (j - (len(systems) - 1) / 2) * w for i in range(len(kinds))]
        ax.bar(xs, vals, width=w, color=colors[s], edgecolor=SURFACE, linewidth=2, label=s)
    ax.set_xticks(range(len(kinds)), labels, fontsize=8.5, color=INK)
    ax.set_ylabel("claims flagged (%)", color=INK2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.axvline(1.5, color=GRID, linewidth=1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "quantstress.png", facecolor=SURFACE, bbox_inches="tight")


def abstention():
    d = json.loads((ROOT / "results" / "delucionqa.json").read_text())
    main = headline(d)
    pts = sorted(main["abstention"], key=lambda r: -r["coverage"])
    xs, ys = [100 * p["coverage"] for p in pts], [100 * p["accuracy"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot(xs, ys, color=SERIES[0], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for x, y in zip(xs, ys, strict=True):
        ax.text(x, y + 1.2, f"{y:.1f}", ha="center", fontsize=8, color=INK)
    ax.invert_xaxis()
    ax.set_xlabel("share of test answers decided (rest sent to review), %", color=INK2, fontsize=9)
    ax.set_ylabel("accuracy on decided answers, %", color=INK2, fontsize=9)
    ax.set_title(f"{main['system']}: abstaining on the closest calls", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(min(ys) - 5, min(100, max(ys) + 6))
    fig.tight_layout()
    fig.savefig(OUT / "abstention.png", facecolor=SURFACE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    delucionqa()
    quantstress()
    abstention()
    print("figures written to", OUT)
