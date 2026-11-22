"""HTTP service: POST /verify with {answer, evidence, question?} -> per-claim verdicts and proofs. GET / serves the UI."""
from __future__ import annotations

import hmac
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from claimgate.pipeline import FULL_MODEL_PATH, MODEL_PATH, ClaimGate

STATIC = Path(__file__).parent / "static"
_MODEL_LOCK = threading.Lock()  # one request at a time on the models (FastAPI runs sync endpoints on a thread pool)
app = FastAPI(title="ClaimGate", version="0.1.0")


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    evidence: str = Field(min_length=1, max_length=40000)
    question: str = Field(default="", max_length=2000)
    policy: str = Field(default="strict", pattern="^(strict|contradictions)$")


@lru_cache(maxsize=1)
def gate() -> ClaimGate:
    """The full model (with the fine-tuned verifier and detectors) when it has been trained, else the shipped zero-shot one."""
    from claimgate.nli import NLI
    model_dir = os.environ.get("CLAIMGATE_NLI")
    nli = NLI(model_dir) if model_dir else NLI()
    path = Path(os.environ.get("CLAIMGATE_MODEL", FULL_MODEL_PATH if FULL_MODEL_PATH.exists() else MODEL_PATH))
    probe = ClaimGate.load(None, path)
    scorers = {}
    if "verifier" in probe.needed_signals:
        from claimgate.verifier import Verifier
        scorers["verifier"] = Verifier()
    if "hhem" in probe.needed_signals:
        from claimgate.detectors import HHEM
        scorers["hhem"] = HHEM()
    if "minicheck" in probe.needed_signals:
        from claimgate.detectors import MiniCheck
        scorers["minicheck"] = MiniCheck()
    return ClaimGate.load(nli, path, scorers)


def check_key(x_api_key: str | None) -> None:
    """Optional shared-secret check: set CLAIMGATE_API_KEY to require the X-API-Key header."""
    expected = os.environ.get("CLAIMGATE_API_KEY")
    if expected and not hmac.compare_digest(expected, x_api_key or ""):
        raise HTTPException(401, "missing or wrong X-API-Key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/verify")
def verify(req: VerifyRequest, x_api_key: str | None = Header(default=None)) -> dict:
    check_key(x_api_key)
    t0 = time.perf_counter()
    try:
        with _MODEL_LOCK:
            report = gate().verify(req.answer, req.evidence, req.question, req.policy)
    except FileNotFoundError as e:
        raise HTTPException(503, f"model files missing: {e}") from e
    out = report.to_dict()
    out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
