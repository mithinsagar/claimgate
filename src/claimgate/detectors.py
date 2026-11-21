"""Two widely used open hallucination detectors, run locally as baselines and as optional signals.

HHEM-2.1-Open (Vectara, Apache-2.0): a Flan-T5-base token classifier over the prompt
"<pad> Determine if the hypothesis is true given the premise?\\n\\nPremise: ...\\n\\nHypothesis: ...", reading P(consistent) at
the first position. Loaded into transformers' T5ForTokenClassification from the published weights, without running the
repository's own code.
MiniCheck-Flan-T5-Large (Tang et al., EMNLP 2024, MIT): document + "</s>" + claim into Flan-T5-large, P(supported) from the
first decoder step over the label tokens 3 (unsupported) and 209 (supported); documents are split into ~500-word chunks
and the best chunk counts, as in the reference implementation.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, T5ForTokenClassification

from claimgate.data import ROOT

HHEM_DIR = ROOT / "models" / "hhem-2.1-open"
FLAN_BASE_DIR = ROOT / "models" / "flan-t5-base"
MINICHECK_DIR = ROOT / "models" / "minicheck-flan-t5-large"
HHEM_PROMPT = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {premise}\n\nHypothesis: {hypothesis}"


def _device() -> str:
    from claimgate.nli import device
    return device()


class HHEM:
    def __init__(self, device: str | None = None, batch_size: int = 8, max_length: int = 2048):
        self.device, self.batch_size, self.max_length = device or _device(), batch_size, max_length
        cfg = AutoConfig.from_pretrained(FLAN_BASE_DIR)
        cfg.num_labels = 2
        self.model = T5ForTokenClassification(cfg)
        state = {k.removeprefix("t5."): v for k, v in load_file(Path(HHEM_DIR) / "model.safetensors").items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or [m for m in missing if "encoder.embed_tokens" not in m]:
            raise RuntimeError(f"HHEM weights did not load cleanly: missing {missing}, unexpected {unexpected}")
        self.model.eval().to(self.device)
        self.tok = AutoTokenizer.from_pretrained(FLAN_BASE_DIR)

    @torch.inference_mode()
    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(consistent) per (premise, hypothesis)."""
        out = np.zeros(len(pairs), dtype=np.float32)
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            texts = [HHEM_PROMPT.format(premise=pairs[i][0], hypothesis=pairs[i][1]) for i in idx]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            logits = self.model(**enc).logits[:, 0, :].float()
            out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
        return out


def _chunks(doc: str, words: int = 500) -> list[str]:
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", doc) if s.strip()]
    out, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > words:
            out.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        out.append(" ".join(cur))
    return out or [doc]


class MiniCheck:
    def __init__(self, device: str | None = None, batch_size: int = 8, max_length: int = 2048):
        self.device, self.batch_size, self.max_length = device or _device(), batch_size, max_length
        self.tok = AutoTokenizer.from_pretrained(MINICHECK_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK_DIR, dtype=torch.float32).eval().to(self.device)

    @torch.inference_mode()
    def _run(self, texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for b in range(0, len(order), self.batch_size):
            idx = order[b:b + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.device)
            dec = torch.zeros((len(idx), 1), dtype=torch.long, device=self.device)
            logits = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, decoder_input_ids=dec).logits[:, 0, :].float()
            out[idx] = torch.softmax(logits[:, [3, 209]], -1)[:, 1].cpu().numpy()
        return out

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(supported) per (document, claim): the best ~500-word chunk counts."""
        texts, owner = [], []
        for i, (doc, claim) in enumerate(pairs):
            for c in _chunks(doc):
                texts.append(c + self.tok.eos_token + claim)
                owner.append(i)
        probs = self._run(texts)
        out = np.zeros(len(pairs), dtype=np.float32)
        for i, p in zip(owner, probs, strict=True):
            out[i] = max(out[i], p)
        return out
