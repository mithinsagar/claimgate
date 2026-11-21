"""HTTP service: POST /verify with {answer, evidence, question?} -> per-claim verdicts and proofs. GET / serves the UI."""
from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from claimgate.pipeline import LLM_MODEL_PATH, MODEL_PATH, ClaimGate

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="ClaimGate", version="0.1.0")


class VerifyRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=8000)
    evidence: str = Field(min_length=1, max_length=40000)
    question: str = Field(default="", max_length=2000)


@lru_cache(maxsize=1)
def gate() -> ClaimGate:
    from claimgate.nli import NLI
    model_dir = os.environ.get("CLAIMGATE_NLI")
    nli = NLI(model_dir) if model_dir else NLI()
    if os.environ.get("CLAIMGATE_USE_LLM") == "1":
        from claimgate.llm import LLMJudge
        return ClaimGate.load(nli, LLM_MODEL_PATH, llm=LLMJudge())
    return ClaimGate.load(nli, Path(os.environ.get("CLAIMGATE_MODEL", MODEL_PATH)))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_judge": os.environ.get("CLAIMGATE_USE_LLM") == "1"}


@app.post("/verify")
def verify(req: VerifyRequest) -> dict:
    t0 = time.perf_counter()
    try:
        report = gate().verify(req.answer, req.evidence, req.question)
    except FileNotFoundError as e:
        raise HTTPException(503, f"model files missing: {e}") from e
    out = report.to_dict()
    out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
