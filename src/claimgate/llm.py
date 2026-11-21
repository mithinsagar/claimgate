"""Per-sentence support probability from a local instruction model (Qwen2.5-7B-Instruct, 4-bit, MLX).

The context, question and full answer are encoded once; the KV cache of that prefix is reused for every sentence of the
answer and trimmed back after each one, so an answer with n sentences costs one long prefill plus n short ones. The
score is P(Yes) / (P(Yes) + P(No)) for "Is this sentence fully supported by the context?" read from the next-token
distribution, so there is no sampling and no output parsing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from claimgate.data import ROOT

MODEL_DIR = ROOT / "models" / "qwen2.5-7b-instruct-4bit"
SYSTEM = "You are a meticulous fact checker for answers generated from {domain}."
INSTRUCTIONS = ("A sentence is supported only if the context states or directly implies everything in it. General knowledge, advice or details "
                "that are not in the context make a sentence unsupported, even if they are true.")


class LLMJudge:
    def __init__(self, model_dir: Path | str = MODEL_DIR, max_context_chars: int = 9000, domain: str = "a car owner's manual"):
        self.system = SYSTEM.format(domain=domain)
        import mlx.core as mx
        from mlx_lm import load
        mx.set_cache_limit(1 << 30)  # MLX keeps freed buffers for reuse; without a cap it grew past 12 GB here
        self.model, self.tok = load(str(model_dir))
        self.max_context_chars = max_context_chars
        self.yes = [self.tok.encode(t)[0] for t in ("Yes", " Yes")]
        self.no = [self.tok.encode(t)[0] for t in ("No", " No")]

    def _forward(self, ids: list[int], cache) -> np.ndarray:
        import mlx.core as mx
        logits = self.model(mx.array(ids)[None], cache=cache)
        last = logits[0, -1].astype(mx.float32)
        mx.eval(last)
        return np.array(last)

    def score(self, context: str, question: str, answer: str, sentences: list[str]) -> list[float]:
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        ctx = context[: self.max_context_chars]
        prefix = (f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{INSTRUCTIONS}\n\nContext:\n{ctx}\n\nQuestion: {question}\n\n"
                  f"Full answer:\n{answer}\n\nSentence to check: ")
        cache = make_prompt_cache(self.model)
        self._forward(self.tok.encode(prefix), cache)
        out = []
        for s in sentences:
            suffix = self.tok.encode(f"{s}\nIs this sentence fully supported by the context? Reply Yes or No.<|im_end|>\n<|im_start|>assistant\n")
            logits = self._forward(suffix, cache)
            ly, ln = np.logaddexp.reduce(logits[self.yes]), np.logaddexp.reduce(logits[self.no])
            out.append(float(1 / (1 + np.exp(ln - ly))))
            trim_prompt_cache(cache, len(suffix))
        import mlx.core as mx
        del cache
        mx.clear_cache()
        return out
