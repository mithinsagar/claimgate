"""ClaimGate end to end: answer + evidence -> one verdict per claim, each with its proof.

Order of authority:
  1. deterministic quantity checks: a number or unit that disagrees with the evidence is CONTRADICTED, with the arithmetic
     as proof, whatever the learned model says;
  2. the calibrated risk model over NLI and grounding features decides SUPPORTED vs NOT_IN_EVIDENCE (or CONTRADICTED when
     the best evidence window contradicts the claim);
  3. claims whose calibrated risk falls inside the abstention band are flagged for review instead of guessed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from claimgate import quantities
from claimgate.baselines import bosch_tokens
from claimgate.data import ROOT, Example
from claimgate.decide import SentenceModel
from claimgate.features import sentence_features

MODEL_PATH = ROOT / "artifacts" / "claimgate_model.joblib"
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+")


def split_claims(answer: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(answer) if len(s.strip()) > 2]


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str                  # SUPPORTED | CONTRADICTED | NOT_IN_EVIDENCE
    risk: float                   # calibrated probability that the claim is not supported by the evidence
    needs_review: bool
    evidence: str
    proof: list[str] = field(default_factory=list)


@dataclass
class Report:
    verdict: str                  # PASS | BLOCK | REVIEW
    answer_risk: float
    claims: list[ClaimVerdict]

    def to_dict(self) -> dict:
        return asdict(self)


class ClaimGate:
    def __init__(self, nli, model: SentenceModel, threshold: float, band: float = 0.1):
        self.nli, self.model, self.threshold, self.band = nli, model, threshold, band

    @classmethod
    def load(cls, nli, path: Path = MODEL_PATH) -> ClaimGate:
        d = joblib.load(path)
        return cls(nli, d["model"], d["threshold"], d.get("band", 0.1))

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "band": self.band}, path)

    def verify(self, answer: str, evidence: str, question: str = "") -> Report:
        claims = split_claims(answer) or [answer.strip()]
        ex = Example(uid="live", split="live", question=question, context=" ".join(bosch_tokens(evidence)), context_cased=evidence,
                     answer=answer, sentences=claims, sentence_labels=["Supported"] * len(claims), label=0, meta={"evidence": evidence})
        df = sentence_features([ex], self.nli)
        risk = self.model.predict_proba(df)
        out = []
        for j, claim in enumerate(claims):
            row = df.iloc[j]
            qv = quantities.check(claim, evidence)
            bad = [v for v in qv if v.status in ("mismatch", "unit_mismatch")]
            _, rewrites = quantities.canonicalize(claim, evidence)
            proof = [v.proof for v in qv] + rewrites
            r = float(risk[j])
            if bad:
                verdict, r = "CONTRADICTED", max(r, 0.99)
            elif r >= self.threshold:
                verdict = "CONTRADICTED" if row.c_at_best > max(0.5, row.e_win_max) else "NOT_IN_EVIDENCE"
                proof.append(f"best evidence window: entailment {row.e_win_max:.2f}, contradiction {row.c_at_best:.2f}")
            else:
                verdict = "SUPPORTED"
                proof.append(f"entailed by the evidence window (p={row.e_win_max:.2f})")
            review = (not bad) and abs(r - self.threshold) < self.band
            out.append(ClaimVerdict(claim, verdict, r, review, str(row.best_evidence), proof))
        answer_risk = float(np.max([c.risk for c in out]))
        if any(c.verdict != "SUPPORTED" and not c.needs_review for c in out):
            overall = "BLOCK"
        elif any(c.needs_review for c in out):
            overall = "REVIEW"
        else:
            overall = "PASS"
        return Report(overall, answer_risk, out)
