"""
Regression tests for answer-quality guardrails.

These sit above the lower-level decoder/model tests and verify that vague or
out-of-scope symptom text does not get forced into a misleading differential.
"""

from chatbot.bot import assess_symptoms


def test_vague_body_part_text_does_not_run_classifier():
    result = assess_symptoms("my elbow feels weird", top_n=3)

    assert result["evidence_bridge"]["mode"] == "free_text_fallback"
    assert result["predictions"][0]["flag"] == "INFO"
    assert result["predictions"][0]["score_type"] == "not run"


def test_urinary_symptom_scope_limitation_blocks_disease_prediction():
    result = assess_symptoms("burning urination with lower back pain for 2 days female age 30", top_n=3)

    assert result["evidence_bridge"]["mode"] == "model_scope_limited"
    assert result["evidence_bridge"]["initial_evidence"] == "E_53"
    assert result["evidence_bridge"]["scope_warning"]["reason"] == "urinary_evidence_outside_model_scope"
    assert result["predictions"][0]["flag"] == "INFO"
    assert result["predictions"][0]["score_type"] == "not run"


def test_red_flag_case_still_runs_when_evidence_is_in_scope():
    result = assess_symptoms("I have chest pain and shortness of breath for 2 days, male age 55", top_n=3)

    assert result["evidence_bridge"]["mode"] == "human_symptoms_to_structured_evidence"
    assert result["evidence_bridge"]["initial_evidence"] == "E_66"
    assert result["red_flag_result"]["has_red_flag"] is True
    assert result["predictions"][0]["score_type"] == "model confidence"
    assert "top_margin" in result["predictions"][0]


def test_assessment_uses_normalized_english_symptoms_for_evidence(monkeypatch):
    def fake_normalize(text):
        return {
            "original_text": text,
            "detected_language": {"language": "marathi_devanagari", "script": "devanagari"},
            "normalized_text": "fever cough",
            "mapped_symptoms": [
                {"source": "fever", "canonical": "fever", "match_type": "ollama_nlu", "score": 1.0},
                {"source": "cough", "canonical": "cough", "match_type": "ollama_nlu", "score": 1.0},
            ],
            "unmapped_terms": [],
            "llm_extraction": {"used": True, "symptoms": ["fever", "cough"], "slots": {}},
        }

    monkeypatch.setattr("chatbot.bot.normalize_symptoms", fake_normalize)

    result = assess_symptoms("मुझे बुखार और खांसी है", top_n=3)
    inferred_codes = {item["code"] for item in result["evidence_bridge"]["inferred_evidences"]}

    assert result["evidence_bridge"]["mode"] == "human_symptoms_to_structured_evidence"
    assert "E_91" in inferred_codes
