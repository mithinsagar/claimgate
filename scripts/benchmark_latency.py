"""Latency of the service pipeline on real DelucionQA test answers, with an empty NLI cache (results/latency.json)."""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.nli import NLI  # noqa: E402
from claimgate.pipeline import FULL_MODEL_PATH, MODEL_PATH, ClaimGate  # noqa: E402
from claimgate.text import context_for_nli  # noqa: E402


def run(path: Path, n: int = 60) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        nli = NLI(cache=Path(tmp) / "cache.sqlite")
        probe = ClaimGate.load(None, path)
        scorers = {}
        if "verifier" in probe.needed_signals:
            from claimgate.verifier import Verifier
            scorers["verifier"] = Verifier()
        for name in ("hhem", "minicheck"):
            if name in probe.needed_signals:
                from claimgate import detectors
                scorers[name] = detectors.HHEM() if name == "hhem" else detectors.MiniCheck()
        gate = ClaimGate.load(nli, path, scorers)
        exs = load_delucionqa("test")[: n + 1]
        gate.verify(exs[0].answer, context_for_nli(exs[0]), exs[0].question)  # warm-up
        per_answer, per_claim = [], []
        for ex in exs[1:]:
            t0 = time.perf_counter()
            rep = gate.verify(ex.answer, context_for_nli(ex), ex.question)
            dt = time.perf_counter() - t0
            per_answer.append(dt)
            per_claim.append(dt / len(rep.claims))
    q = sorted(per_answer)
    return {"answers": n, "median_s": statistics.median(per_answer), "p90_s": q[int(0.9 * len(q))], "median_ms_per_claim": 1000 * statistics.median(per_claim)}


def main() -> None:
    out = {}
    for name, path in (("ClaimGate, no fine-tuned network", MODEL_PATH), ("ClaimGate", FULL_MODEL_PATH)):
        if Path(path).exists():
            out[name] = run(path)
            print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
    (ROOT / "results" / "latency.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
