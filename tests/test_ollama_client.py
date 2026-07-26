"""
tests/test_ollama_client.py
──────────────────────────────
Tests for the grounded-generation verifier added in the refactor. These do
NOT require a running Ollama server — they test verify_grounded_response()
directly, which is the part that matters for safety.
"""

from utils.ollama_client import verify_grounded_response

PAYLOAD = {
    "predictions": [
        {"disease": "Stable angina", "pathology": "Stable angina"},
        {"disease": "Bronchitis", "pathology": "Bronchitis"},
    ]
}


def test_grounded_response_passes():
    text = (
        "Your symptoms were normalized as chest pain and fever. The model "
        "suggests Stable angina or Bronchitis. This is not a diagnosis, "
        "please consult a doctor."
    )
    ok, reason = verify_grounded_response(text, PAYLOAD)
    assert ok is True
    assert reason == ""


def test_dosage_mention_is_rejected():
    text = "You likely have Stable angina. Take 500 mg of aspirin twice daily."
    ok, reason = verify_grounded_response(text, PAYLOAD)
    assert ok is False
    assert "dosage" in reason


def test_ungrounded_disease_is_rejected():
    text = "This strongly suggests Anaphylaxis, please seek care immediately."
    ok, reason = verify_grounded_response(text, PAYLOAD)
    assert ok is False
    assert "Anaphylaxis" in reason


def test_empty_response_is_rejected():
    ok, reason = verify_grounded_response("", PAYLOAD)
    assert ok is False
    assert reason == "empty response"
