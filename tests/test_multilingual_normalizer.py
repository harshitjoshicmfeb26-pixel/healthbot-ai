"""
tests/test_multilingual_normalizer.py
─────────────────────────────────────
Tests for multilingual symptom normalization, including the optional NLU
extractor path.
"""

import pytest

from utils import multilingual_normalizer
from utils.ddxplus_decoder import infer_evidence_codes_from_text, select_initial_evidence


@pytest.mark.parametrize(
    "text,canonical",
    [
        ("coughh", "cough"),
        ("fevr", "fever"),
        ("headeche", "headache"),
        ("nausous", "nausea"),
        ("wheezng", "wheezing"),
    ],
)
def test_validated_typos_use_deterministic_normalization(text, canonical):
    result = multilingual_normalizer.normalize_symptoms(text)

    assert result["normalized_text"] == canonical
    assert result["mapped_symptoms"][0]["match_type"] == "exact"
    assert result["mapped_symptoms"][0]["source"] == text


@pytest.mark.parametrize(
    "text,expected_codes,expected_initial",
    [
        ("coughh and fevr", {"E_201", "E_91"}, "E_201"),
        ("headeche", {"E_53", "E_55_@_V_11"}, "E_53"),
        ("nausous", {"E_148"}, "E_148"),
        ("wheezng", {"E_214"}, "E_214"),
    ],
)
def test_validated_typos_bridge_to_ddxplus_evidence(text, expected_codes, expected_initial):
    matches = infer_evidence_codes_from_text(text, include_denied=True)
    assert {item["code"] for item in matches if not item["negated"]} == expected_codes
    assert select_initial_evidence(matches) == expected_initial


@pytest.mark.parametrize(
    "text,expected_denied",
    [("I do not have coughh", "E_201"), ("no fevr", "E_91"), ("not headeche", "E_53")],
)
def test_validated_typo_normalization_preserves_negation(text, expected_denied):
    matches = infer_evidence_codes_from_text(text, include_denied=True)

    assert expected_denied in {item["code"] for item in matches if item["negated"]}
    assert expected_denied not in {item["code"] for item in matches if not item["negated"]}


@pytest.mark.parametrize("text", ["cold sweats", "pain somewhere", "my chest feels funny"])
def test_validated_typos_do_not_change_unrelated_text(text):
    result = multilingual_normalizer.normalize_symptoms(text)

    assert not any(item.get("match_type") == "exact" and item.get("source") in {
        "coughh", "fevr", "headeche", "nausous", "wheezng"
    } for item in result["mapped_symptoms"])


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


def test_exact_normalization_exposes_source_spans_without_fuzzy_matches():
    matches = multilingual_normalizer.deterministic_exact_matches(
        "mala taap ani khokla aahe"
    )
    assert [(item["source"], item["canonical"]) for item in matches] == [
        ("taap", "fever"),
        ("khokla", "cough"),
    ]
    assert all(item["match_type"] == "exact" for item in matches)
    assert all(isinstance(item["start"], int) for item in matches)
