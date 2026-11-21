"""Baseline: a local LLM (Qwen2.5-7B-Instruct, 4-bit, MLX) judges every answer sentence against the context, zero-shot.

An answer is Hallucinated if the model marks any sentence as not supported. Greedy decoding, no threshold, so dev is not
used. Outputs are cached per answer in artifacts/llm_judge_<split>.jsonl so runs can resume.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimgate.data import ROOT, load_delucionqa  # noqa: E402
from claimgate.text import context_for_nli  # noqa: E402

PROMPT = """You check answers produced by a retrieval-augmented assistant for a car owner's manual.
A sentence is SUPPORTED only if the context states or directly implies everything it says. General knowledge, advice or
details that the context does not contain make a sentence NOT_SUPPORTED, even if they are true.

Context:
{context}

Question: {question}

Answer sentences:
{sentences}

Reply with one line per sentence in the form "<number>: SUPPORTED" or "<number>: NOT_SUPPORTED" and nothing else."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-7b-instruct-4bit"))
    args = ap.parse_args()
    from mlx_lm import generate, load
    model, tok = load(args.model)
    out_path = ROOT / "artifacts" / f"llm_judge_{args.split}.jsonl"
    done = {json.loads(line)["uid"] for line in open(out_path)} if out_path.exists() else set()
    examples = [e for e in load_delucionqa(args.split) if e.uid not in done]
    t0 = time.time()
    with open(out_path, "a") as f:
        for k, ex in enumerate(examples):
            sents = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(ex.sentences))
            msg = PROMPT.format(context=context_for_nli(ex), question=ex.question, sentences=sents)
            prompt = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
            text = generate(model, tok, prompt=prompt, max_tokens=12 * len(ex.sentences) + 20, verbose=False)
            marks = {int(m.group(1)): m.group(2) for m in re.finditer(r"(\d+)\s*[:.)-]\s*(NOT_SUPPORTED|SUPPORTED)", text)}
            pred = [int(marks.get(i + 1, "NOT_SUPPORTED") == "NOT_SUPPORTED") for i in range(len(ex.sentences))]
            f.write(json.dumps({"uid": ex.uid, "pred_sentences": pred, "parsed": len(marks), "raw": text}) + "\n")
            f.flush()
            if k % 25 == 0:
                print(f"{k}/{len(examples)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
