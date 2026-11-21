"""ClaimGate end to end: answer + evidence -> one verdict per claim, each with its proof.

Order of authority:
  1. deterministic quantity checks: a number or unit that disagrees with the evidence is CONTRADICTED, with the arithmetic
     as proof, whatever the learned model says; the contradiction gate does the same for claims whose wording matches the
     evidence but whose meaning NLI finds contradicted (a flipped "not", "before" instead of "after");
  2. the calibrated risk model over NLI and grounding features decides SUPPORTED vs NOT_IN_EVIDENCE (or CONTRADICTED when
     the best evidence window contradicts the claim);
  3. claims whose risk is within the review band of the threshold (logit distance, sized on dev so about 20% of
     answers go to review) are flagged for review instead of guessed.
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
from claimgate.decide import SentenceModel, contradiction_gate, margin
from claimgate.features import sentence_features
from claimgate.text import sentences

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
    def __init__(self, nli, model: SentenceModel, threshold: float, band: float = 0.5, llm=None):
        self.nli, self.model, self.threshold, self.band, self.llm = nli, model, threshold, band, llm
        if "llm_yes" in model.features and llm is None and nli is not None:
            raise ValueError("this model uses the LLM judge feature; pass llm=LLMJudge()")

    @classmethod
    def load(cls, nli, path: Path = MODEL_PATH, llm=None) -> ClaimGate:
        d = joblib.load(path)
        return cls(nli, d["model"], d["threshold"], d.get("band", 0.5), llm)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold, "band": self.band}, path)

    def verify(self, answer: str, evidence: str, question: str = "") -> Report:
        claims = split_claims(answer) or [answer.strip()]
        ex = Example(uid="live", split="live", question=question, context=" ".join(bosch_tokens(evidence)), context_cased=evidence,
                     answer=answer, sentences=claims, sentence_labels=["Supported"] * len(claims), label=0, meta={"evidence": evidence})
        df = sentence_features([ex], self.nli)
        if "llm_yes" in self.model.features:
            df["llm_yes"] = self.llm.score(evidence, question, answer, claims)
        risk = self.model.predict_proba(df)
        gate = contradiction_gate(df)
        out = []
        for j, claim in enumerate(claims):
            row = df.iloc[j]
            qv = quantities.check(claim, evidence)
            bad = [v for v in qv if v.status in ("mismatch", "unit_mismatch")]
            proof = [v.proof for v in qv]
            r = float(risk[j])
            if bad:
                verdict, r = "CONTRADICTED", max(r, 0.99)
            elif gate[j]:
                verdict, r = "CONTRADICTED", max(r, 0.99)
                proof.append(f"an evidence window contradicts the claim (p={row.c_win_max:.2f}) and none entails it "
                             f"(p={max(row.e_win_max, row.e_chunk_max):.2f}) although its wording comes from the evidence")
            elif r >= self.threshold:
                verdict = "CONTRADICTED" if row.c_at_best > max(0.5, row.e_win_max) else "NOT_IN_EVIDENCE"
                proof.append(f"best evidence window: entailment {row.e_win_max:.2f}, contradiction {row.c_at_best:.2f}")
            else:
                verdict = "SUPPORTED"
                proof.append(f"entailed by the evidence window (p={row.e_win_max:.2f})")
            review = (not bad) and (not gate[j]) and float(margin(r, self.threshold)) < self.band
            shown = str(row.best_evidence)
            if bad:  # show the evidence sentence holding the conflicting number, not the best-entailment window
                shown = next((x for x in sentences(evidence) if bad[0].against and bad[0].against in x), shown)
            out.append(ClaimVerdict(claim, verdict, r, review, shown, proof))
        answer_risk = float(np.max([c.risk for c in out]))
        if any(c.verdict != "SUPPORTED" and not c.needs_review for c in out):
            overall = "BLOCK"
        elif any(c.needs_review for c in out):
            overall = "REVIEW"
        else:
            overall = "PASS"
        return Report(overall, answer_risk, out)
