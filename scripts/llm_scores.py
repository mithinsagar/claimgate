"""Score every DelucionQA answer sentence with the local LLM judge (resumable; artifacts/llm/<split>.jsonl)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.llm import LLMJudge  # noqa: E402
from claimgate.text import context_for_nli  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["dev", "test", "train"])
    args = ap.parse_args()
    judge = LLMJudge()
    out_dir = ROOT / "artifacts" / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        path = out_dir / f"{split}.jsonl"
        done = {json.loads(line)["uid"] for line in open(path)} if path.exists() else set()
        todo = [e for e in load_delucionqa(split) if e.uid not in done]
        t0 = time.time()
        with open(path, "a") as f:
            for k, ex in enumerate(todo):
                p = judge.score(context_for_nli(ex), ex.question, ex.answer, ex.sentences)
                f.write(json.dumps({"uid": ex.uid, "p_yes": p}) + "\n")
                f.flush()
                if k % 100 == 0:
                    print(f"{split} {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        print(f"{split} done", flush=True)


if __name__ == "__main__":
    main()
