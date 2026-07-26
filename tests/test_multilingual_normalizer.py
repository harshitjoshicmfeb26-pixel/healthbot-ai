"""
tests/test_multilingual_normalizer.py
─────────────────────────────────────
Tests for multilingual symptom normalization, including the optional NLU
extractor path.
"""

from utils import multilingual_normalizer


def test_normalizer_can_use_ollama_nlu_fallback(monkeypatch):
    def fake_extract(text):
        return {
            "enabled": True,
            "used": True,
            "model": "qwen3.5:9b",
            "symptoms": ["fever", "cough"],
            "slots": {"age": 35, "gender": "M"},
            "language": "hindi",
        }

    monkeypatch.setattr(multilingual_normalizer, "extract_clinical_details", fake_extract)

    result = multilingual_normalizer.normalize_symptoms("मुझे बुखार और खांसी है")

    assert result["normalized_text"] == "fever cough"
    assert result["llm_extraction"]["slots"]["age"] == 35
    assert {item["canonical"] for item in result["mapped_symptoms"]} == {"fever", "cough"}
    assert {item["match_type"] for item in result["mapped_symptoms"]} == {"ollama_nlu"}
