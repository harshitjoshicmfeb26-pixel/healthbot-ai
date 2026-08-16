"""
tests/test_ddxplus_decoder.py
───────────────────────────────
Tests for utils.ddxplus_decoder against the bundled demo metadata subset
(data/release_evidences.json, data/release_conditions.json). These are NOT
the official DDXPlus files — see data/README_DEMO_DATA.md — but they are
real enough in structure to exercise decode_evidence / decode_condition /
condition_severity / infer_evidence_codes_from_text meaningfully.
"""

import pytest

from utils.ddxplus_decoder import (
    condition_severity,
    decode_condition,
    decode_evidence,
    infer_evidence_codes_from_text,
    select_initial_evidence,
)


def test_decode_evidence_binary():
    decoded = decode_evidence("E_91")
    assert decoded["known"] is True
    assert decoded["base_code"] == "E_91"
    assert "fever" in decoded["meaning"].lower()


def test_decode_evidence_categorical_with_value():
    decoded = decode_evidence("E_55_@_V_29")
    assert decoded["known"] is True
    assert decoded["base_code"] == "E_55"
    assert decoded["value_code"] == "V_29"
    assert "lower chest" in decoded["meaning"].lower()


def test_decode_evidence_unknown_code_degrades_gracefully():
    decoded = decode_evidence("E_99999")
    assert decoded["known"] is False
    assert "unknown" in decoded["meaning"].lower()


def test_decode_condition_known():
    decoded = decode_condition("Anaphylaxis")
    assert decoded["known"] is True
    assert decoded["display_name"] == "Anaphylaxis"
    assert decoded["severity"] == 1


def test_decode_condition_unknown_degrades_gracefully():
    decoded = decode_condition("Not A Real Disease")
    assert decoded["known"] is False
    assert decoded["severity"] is None


def test_condition_severity_helper():
    assert condition_severity("Myasthenia gravis") == 3
    assert condition_severity("URTI") == 5
    assert condition_severity("Not A Real Disease") is None


def test_infer_evidence_codes_positive_match():
    matches = infer_evidence_codes_from_text("I have chest pain and a fever")
    codes = {m["code"] for m in matches}
    assert "E_53" in codes
    assert "E_91" in codes
    assert all(m["negated"] is False for m in matches)


def test_infer_evidence_codes_excludes_negated_by_default():
    matches = infer_evidence_codes_from_text("I have no chest pain but I have a fever")
    codes = {m["code"] for m in matches}
    assert "E_91" in codes
    assert "E_53" not in codes  # denied, excluded by default


def test_infer_evidence_codes_include_denied_flag():
    matches = infer_evidence_codes_from_text(
        "I have no chest pain but I have a fever", include_denied=True
    )
    negated_codes = {m["code"] for m in matches if m["negated"]}
    affirmed_codes = {m["code"] for m in matches if not m["negated"]}
    assert "E_53" in negated_codes
    assert "E_91" in affirmed_codes


def test_location_only_word_does_not_become_pain_evidence():
    matches = infer_evidence_codes_from_text("my elbow feels weird")
    assert matches == []


def test_location_with_pain_context_is_marked_weak():
    matches = infer_evidence_codes_from_text("my elbow hurts")
    assert matches
    assert {m["evidence_strength"] for m in matches} == {"location_with_context"}


def test_hinglish_headache_sore_throat_fever_aliases():
    matches = infer_evidence_codes_from_text("sar dard ho raha, gaka kharab hai, fever bhi hai")
    codes = {m["code"] for m in matches}
    assert "E_55_@_V_11" in codes
    assert "E_97" in codes
    assert "E_91" in codes


def test_select_initial_evidence_prefers_specific_code_over_generic_pain():
    matches = infer_evidence_codes_from_text("burning urination with lower back pain")
    assert select_initial_evidence(matches) == "E_55_@_V_185"


@pytest.mark.parametrize(
    "text,expected_codes",
    [
        ("mala taap ani khokla aahe", {"E_91", "E_201"}),
        ("mala taap aahe", {"E_91"}),
        ("mujhe bukhar aur khansi hai", {"E_91", "E_201"}),
        ("मला ताप आणि खोकला आहे", {"E_91", "E_201"}),
        ("I have fever and cough", {"E_91", "E_201"}),
    ],
)
def test_deterministic_canonical_normalization_bridges_to_ddx_codes(text, expected_codes):
    matches = infer_evidence_codes_from_text(text)
    codes = {item["code"] for item in matches}
    assert expected_codes <= codes
    assert len(codes) == len(matches)


def test_canonical_bridge_preserves_original_negation():
    matches = infer_evidence_codes_from_text("mala taap nahi", include_denied=True)
    assert "E_91" not in {item["code"] for item in matches if not item["negated"]}
    assert "E_91" in {item["code"] for item in matches if item["negated"]}


def test_canonical_bridge_preserves_structured_chest_pain_codes():
    matches = infer_evidence_codes_from_text("chest pain")
    assert {item["code"] for item in matches} >= {"E_53", "E_55_@_V_29"}


def test_canonical_bridge_does_not_create_new_nested_context_codes():
    before = infer_evidence_codes_from_text("family asthma")
    after_codes = {item["code"] for item in before}
    assert "E_87" in after_codes


@pytest.mark.parametrize(
    "text,expected_codes,forbidden_codes",
    [
        ("family asthma", {"E_87"}, {"E_124"}),
        ("family asthma and cough", {"E_87", "E_201"}, {"E_124"}),
        ("I have asthma and my mother has asthma", {"E_124", "E_142"}, set()),
        ("previous diabetes", {"E_69"}, set()),
        ("family asthma and diabetes history", {"E_87", "E_69"}, {"E_124"}),
        ("no family history of asthma", set(), {"E_87", "E_124"}),
        ("my family has asthma but I don't", {"E_87"}, {"E_124"}),
    ],
)
def test_contextual_alias_precedence(text, expected_codes, forbidden_codes):
    matches = infer_evidence_codes_from_text(text, include_denied=True)
    positive = {item["code"] for item in matches if not item["negated"]}
    assert expected_codes <= positive
    assert forbidden_codes.isdisjoint(positive)


def test_chest_pain_multi_code_mapping_is_preserved_under_precedence():
    for text in ("chest pain", "severe chest pain"):
        codes = {item["code"] for item in infer_evidence_codes_from_text(text)}
        assert {"E_53", "E_55_@_V_29"} <= codes
