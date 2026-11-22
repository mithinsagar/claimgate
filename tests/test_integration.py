"""End-to-end test with real model weights (CI job "integration"; skipped unless CLAIMGATE_INTEGRATION=1).

The shipped decision model runs on top of a tiny real NLI model (cross-encoder/nli-deberta-v3-xsmall, so the job stays
small), the real HHEM-2.1-Open weights and the real bge-small embeddings, through the same code path as the service.
Scores from the tiny NLI model differ from DeBERTa-v3-large, so the test checks wiring and the deterministic layers,
not accuracy.
"""
import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(os.environ.get("CLAIMGATE_INTEGRATION") != "1", reason="needs model weights; set CLAIMGATE_INTEGRATION=1")

EVIDENCE = ("Wheel and tire care. Tighten the lug nuts in a star pattern to 130 ft-lb (176 N·m). Check the torque again after the first "
            "25 miles. Tire pressure should be checked at least once a month when the tires are cold. The spare tire is for temporary use "
            "only and should not be driven above 50 mph.")
ANSWER = ("Tighten the lug nuts in a star pattern to 176 N·m. The spare tire can be driven at up to 70 mph. "
          "Rotate your tires every 5,000 miles to extend their life.")


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("CLAIMGATE_NLI", "cross-encoder/nli-deberta-v3-xsmall")
    from claimgate import serve
    serve.gate.cache_clear()
    return TestClient(serve.app)


def test_verify_end_to_end_with_real_models(client):
    r = client.post("/verify", json={"answer": ANSWER, "evidence": EVIDENCE})
    assert r.status_code == 200, r.text
    body = r.json()
    claims = {c["claim"]: c for c in body["claims"]}
    assert len(claims) == 3 and body["verdict"] == "BLOCK"
    converted = claims["Tighten the lug nuts in a star pattern to 176 N·m."]
    assert any("after unit conversion" in p for p in converted["proof"])
    speed = claims["The spare tire can be driven at up to 70 mph."]
    assert speed["verdict"] == "CONTRADICTED" and "50 mph" in speed["evidence"]
    rotate = claims["Rotate your tires every 5,000 miles to extend their life."]
    assert rotate["verdict"] != "SUPPORTED"
    assert rotate["risk"] > converted["risk"]


def test_contradictions_policy_with_real_models(client):
    r = client.post("/verify", json={"answer": "Rotate your tires every 5,000 miles to extend their life.", "evidence": EVIDENCE,
                                     "policy": "contradictions"})
    assert r.status_code == 200
    assert r.json()["verdict"] in ("PASS", "REVIEW") and r.json()["warnings"] <= 1
