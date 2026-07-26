"""
tests/test_ollama_nlu_extractor.py
──────────────────────────────────
Tests for the optional Ollama NLU JSON extractor. These tests mock Ollama; no
local server or model is required.
"""

from types import SimpleNamespace

from utils import ollama_nlu_extractor as extractor


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "response": """
            {
              "symptoms": ["fever", "khansi", "Tuberculosis"],
              "age": "35 years old",
              "gender": "male",
              "duration": "2 days",
              "severity": "8/10",
              "pain_location": "head",
              "previous_disease": "diabetes",
              "family_history": "none",
              "language": "hinglish"
            }
            """
        }


def test_ollama_nlu_extracts_valid_symptoms_and_slots(monkeypatch):
    def fake_post(*args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr(extractor, "OLLAMA_NLU_ENABLED", True)
    monkeypatch.setattr(extractor, "requests", SimpleNamespace(post=fake_post))

    result = extractor.extract_clinical_details("mujhe bukhar aur khansi hai")

    assert result["used"] is True
    assert result["symptoms"] == ["fever", "cough"]
    assert "Tuberculosis" not in result["symptoms"]
    assert result["slots"]["age"] == 35
    assert result["slots"]["gender"] == "M"
    assert result["slots"]["duration"] == "2 days"
    assert result["slots"]["severity"] == "8/10"
