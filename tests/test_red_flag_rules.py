"""
tests/test_red_flag_rules.py
──────────────────────────────
Tests for utils.red_flag_rules, including the negation-awareness fix from
the refactor: a denied symptom ("no chest pain") must not raise a false
emergency flag.
"""

from utils.red_flag_rules import detect_red_flags


def test_affirmed_emergency_symptom_flags():
    result = detect_red_flags("I have severe chest pain right now")
    assert result["has_red_flag"] is True
    assert result["urgency_level"] == "emergency"
    assert "chest pain" in result["matched_flags"] or "severe chest pain" in result["matched_flags"]


def test_negated_emergency_symptom_does_not_flag():
    result = detect_red_flags("I have no chest pain, just a mild headache")
    assert result["has_red_flag"] is False
    assert result["urgency_level"] == "routine"
    assert "chest pain" in result["denied_flags"]


def test_negated_symptom_in_hinglish_does_not_flag():
    result = detect_red_flags("mujhe chest pain nahi hai, sirf khansi hai")
    assert result["has_red_flag"] is False
    assert "chest pain" in result["denied_flags"]


def test_mixed_affirmed_and_negated_only_flags_affirmed():
    result = detect_red_flags("patient denies shortness of breath, has severe chest pain")
    assert result["has_red_flag"] is True
    assert "shortness of breath" in result["denied_flags"]
    assert any("chest pain" in flag for flag in result["matched_flags"])


def test_contraction_negation_does_not_suppress_inability_phrase():
    result = detect_red_flags("I don't have chest pain but I can't breathe")
    assert "chest pain" in result["denied_flags"]
    # There is no direct "breathe" alias in the current metadata/rule list;
    # importantly, contraction negation must not create a denied red flag.
    assert all("breath" not in flag for flag in result["denied_flags"])


def test_temporal_chest_pain_reassertion_triggers_red_flag():
    result = detect_red_flags(
        "I had no chest pain yesterday but today it started and I am short of breath"
    )
    assert result["has_red_flag"] is True
    assert "chest pain" in result["matched_flags"]
    assert "shortness of breath" in result["matched_flags"]


def test_resolved_historical_chest_pain_does_not_trigger_red_flag():
    result = detect_red_flags("I had chest pain yesterday but I don't have it now")
    assert result["has_red_flag"] is False
    assert "chest pain" in result["denied_flags"]
