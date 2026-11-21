"""Pipeline and API tests with a stub NLI model, so they run without downloading weights."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from claimgate import serve
from claimgate.decide import SentenceModel, answer_scores, tune_threshold
from claimgate.pipeline import ClaimGate, split_claims
from claimgate.text import content_words


class StubNLI:
    """Entailment = share of the hypothesis' content words found in the premise."""

    def score(self, pairs):
        out = []
        for p, h in pairs:
            hw, pw = set(content_words(h)), set(content_words(p))
            e = len(hw & pw) / max(1, len(hw))
            out.append([e, 1 - e, 0.0])
        return np.array(out, dtype=np.float32)


def toy_model() -> SentenceModel:
    """Risk model on the two NLI features only, so verdicts depend on the stub's entailment and nothing else."""
    rng = np.random.default_rng(0)
    n = 400
    feats = ["e_win_max", "c_at_best"]
    df = pd.DataFrame(rng.random((n, 2)), columns=feats)
    df["target"] = (df.e_win_max < 0.5).astype(int)
    df["uid"] = [f"a{i // 2}" for i in range(n)]
    return SentenceModel.fit(df, feats, seeds=1, folds=3)


@pytest.fixture(scope="module")
def gate():
    return ClaimGate(StubNLI(), toy_model(), threshold=0.5, band=0.05)


EVIDENCE = "Tighten the lug nuts to 130 ft-lb. Check the tire pressure once a month when the tires are cold. Do not drive the spare above 50 mph."


def test_split_claims():
    assert split_claims("One claim here. Another one! A third?") == ["One claim here.", "Another one!", "A third?"]


def test_supported_and_contradicted(gate):
    rep = gate.verify("Check the tire pressure once a month when the tires are cold. Tighten the lug nuts to 200 ft-lb.", EVIDENCE)
    assert [c.verdict for c in rep.claims] == ["SUPPORTED", "CONTRADICTED"]
    assert rep.verdict == "BLOCK"
    assert any("difference" in p for p in rep.claims[1].proof)
    assert rep.claims[1].evidence == "Tighten the lug nuts to 130 ft-lb."


def test_converted_value_is_not_contradicted(gate):
    rep = gate.verify("Tighten the lug nuts to 176 N·m.", EVIDENCE)
    assert rep.claims[0].verdict == "SUPPORTED"
    assert any("after unit conversion" in p for p in rep.claims[0].proof)


def test_unsupported_claim(gate):
    rep = gate.verify("Rotate tires every 5,000 miles for longer tread life.", EVIDENCE)
    assert rep.claims[0].verdict != "SUPPORTED"


def test_answer_level_max_and_threshold():
    df = pd.DataFrame({"uid": ["a", "a", "b"], "answer_label": [1, 1, 0]})
    a = answer_scores(df, np.array([0.1, 0.9, 0.2]))
    assert a.loc["a", "score"] == 0.9 and a.loc["b", "score"] == 0.2
    t, f = tune_threshold(a.label.to_numpy(), a.score.to_numpy())
    assert f == 1.0 and 0.2 < t <= 0.9


def test_api(gate, monkeypatch):
    monkeypatch.setattr(serve, "gate", lambda: gate)
    client = TestClient(serve.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/verify", json={"answer": "Do not drive the spare above 50 mph.", "evidence": EVIDENCE})
    assert r.status_code == 200 and r.json()["claims"][0]["verdict"] == "SUPPORTED"
    assert client.post("/verify", json={"answer": "", "evidence": EVIDENCE}).status_code == 422
    assert "ClaimGate" in client.get("/").text


def test_contradiction_gate_needs_grounded_wording_and_no_entailment():
    from claimgate.decide import contradiction_gate
    df = pd.DataFrame({"c_win_max": [0.97, 0.97, 0.97, 0.5], "e_win_max": [0.01, 0.6, 0.01, 0.01], "e_chunk_max": [0.02, 0.0, 0.02, 0.02],
                       "novel_frac": [0.0, 0.0, 0.5, 0.0]})
    assert contradiction_gate(df).tolist() == [True, False, False, False]


def test_contradictions_policy_only_blocks_contradictions(gate):
    answer = "Check the tire pressure once a month when the tires are cold. Rotate tires every 5,000 miles for longer tread life."
    strict = gate.verify(answer, EVIDENCE, policy="strict")
    lenient = gate.verify(answer, EVIDENCE, policy="contradictions")
    assert strict.verdict == "BLOCK"
    assert lenient.verdict == "PASS" and lenient.warnings == 1
    assert gate.verify("Tighten the lug nuts to 200 ft-lb.", EVIDENCE, policy="contradictions").verdict == "BLOCK"


def test_api_key_is_enforced_when_set(gate, monkeypatch):
    monkeypatch.setattr(serve, "gate", lambda: gate)
    monkeypatch.setenv("CLAIMGATE_API_KEY", "s3cret")
    client = TestClient(serve.app)
    body = {"answer": "Do not drive the spare above 50 mph.", "evidence": EVIDENCE}
    assert client.post("/verify", json=body).status_code == 401
    assert client.post("/verify", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.post("/verify", json={**body, "policy": "lenient"}).status_code in (401, 422)
