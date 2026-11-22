"""Cross-encoder NLI scoring with an on-disk cache (premise, hypothesis -> entailment / neutral / contradiction)."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from claimgate.data import ROOT

DEFAULT_MODEL = ROOT / "models" / "deberta-v3-large-nli"


def device() -> str:
    if os.environ.get("CLAIMGATE_DEVICE"):
        return os.environ["CLAIMGATE_DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


class NLI:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL, batch_size: int = 16, max_length: int = 512, cache: Path | None = None,
                 verbose: bool = False):
        self.model_dir, self.verbose = Path(model_dir), verbose
        self.batch_size, self.max_length = batch_size, max_length
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).eval().to(device())
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.order = [labels["entailment"], labels.get("neutral", labels.get("not_entailment")), labels.get("contradiction", labels.get("not_entailment"))]
        cache = cache or ROOT / "artifacts" / f"nli_cache_{self.model_dir.name}.sqlite"
        cache.parent.mkdir(parents=True, exist_ok=True)
        # the service answers requests on a thread pool: one connection shared across threads, guarded by a lock
        self.db = sqlite3.connect(cache, check_same_thread=False)
        self.db.execute("create table if not exists s (k text primary key, e real, n real, c real)")
        self._lock = threading.Lock()

    @staticmethod
    def _key(p: str, h: str) -> str:
        return hashlib.sha1((p + "\x00" + h).encode()).hexdigest()

    @torch.inference_mode()
    def _run(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out, t0 = [], time.time()
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for b in range(0, len(order), self.batch_size):
            if self.verbose and b and b % (self.batch_size * 200) == 0:
                print(f"  nli {b}/{len(pairs)} pairs, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            idx = order[b:b + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first", max_length=self.max_length,
                           padding=True, return_tensors="pt").to(self.model.device)
            probs = torch.softmax(self.model(**enc).logits.float(), -1)[:, self.order].cpu().numpy()
            out.extend(zip(idx, probs, strict=True))
        res = np.zeros((len(pairs), 3), dtype=np.float32)
        for i, p in out:
            res[i] = p
        return res

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Rows of [P(entailment), P(neutral), P(contradiction)]."""
        with self._lock:
            return self._score(pairs)

    def _score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        keys = [self._key(p, h) for p, h in pairs]
        found = {}
        for b in range(0, len(keys), 900):
            chunk = keys[b:b + 900]
            q = f"select k, e, n, c from s where k in ({','.join('?' * len(chunk))})"
            found.update({k: (e, n, c) for k, e, n, c in self.db.execute(q, chunk)})
        todo = sorted({i for i, k in enumerate(keys) if k not in found}, key=lambda i: keys[i])
        uniq = {}
        for i in todo:
            uniq.setdefault(keys[i], pairs[i])
        if uniq:
            ks = list(uniq)
            probs = self._run([uniq[k] for k in ks])
            self.db.executemany("insert or replace into s values (?,?,?,?)", [(k, *map(float, p)) for k, p in zip(ks, probs, strict=True)])
            self.db.commit()
            found.update({k: tuple(p) for k, p in zip(ks, probs, strict=True)})
        return np.array([found[k] for k in keys], dtype=np.float32)
