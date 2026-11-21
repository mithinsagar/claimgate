"""Evaluate the deterministic quantity checker against human hallucination spans in RAGTruth (all three task types).

Precision: of response sentences the checker marks as contradicting the source (value or unit mismatch), the share that
overlap a span annotators marked as hallucinated.
Recall: of annotated hallucination spans that contain a number, the share whose sentence the checker flags.
No model is involved and nothing is tuned on RAGTruth, so every split can be used.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate import quantities  # noqa: E402
from claimgate.data import ROOT  # noqa: E402
from claimgate.ragtruth import RT, _sentences_with_offsets  # noqa: E402


def source_text(src: dict) -> str:
    info = src["source_info"]
    if src["task_type"] == "QA":
        return re.sub(r"passage \d+:", "", info["passages"])
    return info if isinstance(info, str) else json.dumps(info)


def main() -> None:
    sources = {s["source_id"]: s for s in map(json.loads, open(RT / "source_info.jsonl", encoding="utf-8"))}
    stats = Counter()
    examples = []
    for r in map(json.loads, open(RT / "response.jsonl", encoding="utf-8")):
        src = sources[r["source_id"]]
        task = src["task_type"]
        evidence = source_text(src)
        spans = [(lab["start"], lab["end"], lab["text"]) for lab in r["labels"]]
        flagged_ranges = []
        for sent, s, e in _sentences_with_offsets(r["response"]):
            verdicts = quantities.check(sent, evidence)
            bad = [v for v in verdicts if v.status in ("mismatch", "unit_mismatch")]
            if bad:
                hit = any(s < e2 and s2 < e for s2, e2, _ in spans)
                strength = max(v.overlap for v in bad)
                for k in (1, 2, 3):
                    if strength >= k:
                        stats[(r["split"], f"ov{k}")] += 1
                        stats[(r["split"], f"ov{k}_hit")] += hit
                stats[(task, "flagged")] += 1
                stats[(task, "flagged_hit")] += hit
                flagged_ranges.append((s, e))
                if len(examples) < 400:
                    examples.append({"task": task, "sentence": sent, "human_span": hit,
                                     "proof": [v.proof for v in verdicts if v.status in ("mismatch", "unit_mismatch")]})
        for s2, e2, text in spans:
            if re.search(r"\d", text):
                stats[(task, "numeric_spans")] += 1
                stats[(task, "numeric_spans_caught")] += any(s < e2 and s2 < e for s, e in flagged_ranges)
    by_overlap = {}
    for split in ("train", "test"):
        by_overlap[split] = {k: {"flagged": stats[(split, f"ov{k}")],
                                 "precision": stats[(split, f"ov{k}_hit")] / max(1, stats[(split, f"ov{k}")])} for k in (1, 2, 3)}
        print(split, {k: (v["flagged"], round(v["precision"], 3)) for k, v in by_overlap[split].items()})
    out = {}
    for task in ("QA", "Summary", "Data2txt"):
        f, h = stats[(task, "flagged")], stats[(task, "flagged_hit")]
        n, c = stats[(task, "numeric_spans")], stats[(task, "numeric_spans_caught")]
        out[task] = {"flagged_sentences": f, "precision": h / max(1, f), "numeric_hallucination_spans": n, "recall": c / max(1, n)}
        print(f"{task:9s} flagged {f:5d}  precision {h / max(1, f):.3f} | numeric hallucination spans {n:5d}  caught {c / max(1, n):.3f}")
    report = {"by_task": out, "by_context_overlap": by_overlap, "sample": examples[:60]}
    (ROOT / "results" / "quantity_checker_ragtruth.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
