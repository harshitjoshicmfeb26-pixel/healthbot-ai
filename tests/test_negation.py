"""
tests/test_negation.py
───────────────────────
Unit tests for utils.negation — the NegEx-style negation detector added in
the refactor (see docs/CHANGELOG_REFACTOR.md).
"""

import pytest

from utils.ddxplus_decoder import infer_evidence_codes_from_text
from utils.negation import (
    filter_negated,
    find_phrase_span,
    is_negated,
    temporal_reassertion,
    temporal_resolution,
)


@pytest.mark.parametrize(
    "text,phrase,expect_negated",
    [
        ("I have no chest pain but I do have a headache", "chest pain", True),
        ("I have no chest pain but I do have a headache", "headache", False),
        ("patient denies fever, has cough", "fever", True),
        ("patient denies fever, has cough", "cough", False),
        ("chest pain was ruled out, abdominal pain confirmed", "chest pain", True),
        ("chest pain was ruled out, abdominal pain confirmed", "abdominal pain", False),
        ("no fever, no cough, no headache", "headache", True),
        ("severe chest pain right now", "chest pain", False),
    ],
)
def test_english_negation(text, phrase, expect_negated):
    span = find_phrase_span(text, phrase)
    assert span is not None
    assert is_negated(text, span[0], span[1]) is expect_negated


def test_pseudo_negation_is_not_treated_as_negation():
    # "not ruled out" means the opposite of "ruled out" — classic NegEx trap.
    text = "MI was not ruled out, patient stable"
    span = find_phrase_span(text, "MI")
    assert span is not None
    assert is_negated(text, span[0], span[1]) is False


@pytest.mark.parametrize(
    "text,phrase,expect_negated",
    [
        ("mujhe bukhar nahi hai par khansi hai", "bukhar", True),
        ("mujhe bukhar nahi hai par khansi hai", "khansi", False),
        ("मला ताप नाही पण खोकला आहे", "ताप", True),
        ("मला ताप नाही पण खोकला आहे", "खोकला", False),
    ],
)
def test_hinglish_and_marathi_post_negation(text, phrase, expect_negated):
    span = find_phrase_span(text, phrase)
    assert span is not None
    assert is_negated(text, span[0], span[1]) is expect_negated


def test_filter_negated_splits_affirmed_and_denied():
    text = "I have no chest pain but I do have a headache and mild fever"
    affirmed, denied = filter_negated(text, ["chest pain", "headache", "fever"])
    assert denied == ["chest pain"]
    assert set(affirmed) == {"headache", "fever"}


def test_empty_text_has_no_negation():
    assert is_negated("", 0, 0) is False


@pytest.mark.parametrize(
    "text,denied_codes,affirmed_codes",
    [
        ("I don't have fever", {"E_91"}, set()),
        ("I don't have fever or cough", {"E_91", "E_201"}, set()),
        ("I don't have fever but I have cough", {"E_91"}, {"E_201"}),
        ("I don't have fever and nausea, but I do have cough", {"E_91"}, {"E_201"}),
        ("I haven't had fever", {"E_91"}, set()),
        ("I wasn't experiencing cough", {"E_201"}, set()),
        ("It isn't chest pain", {"E_53", "E_55_@_V_29"}, set()),
        ("I don’t have fever", {"E_91"}, set()),
    ],
)
def test_contraction_negation_is_scoped_to_symptoms(text, denied_codes, affirmed_codes):
    matches = infer_evidence_codes_from_text(text, include_denied=True)
    denied = {item["code"] for item in matches if item["negated"]}
    affirmed = {item["code"] for item in matches if not item["negated"]}
    assert denied_codes <= denied
    assert denied_codes.isdisjoint(affirmed)
    assert affirmed_codes <= affirmed


def test_contraction_pseudo_negation_remains_positive():
    matches = infer_evidence_codes_from_text(
        "Not only cough, I also have fever", include_denied=True
    )
    assert {item["code"] for item in matches if item["negated"]} == set()
    assert {item["code"] for item in matches if not item["negated"]} >= {"E_91", "E_201"}


def test_inability_contractions_are_not_absence_claims():
    # The current metadata does not expose a direct breathing-difficulty alias;
    # this guard still verifies that an inability phrase is never marked denied
    # solely because it contains "can't" or "cannot".
    for text in ("I can't breathe", "I cannot breathe properly", "I couldn't walk"):
        span = find_phrase_span(text, "breathe") or find_phrase_span(text, "walk")
        assert span is not None
        assert is_negated(text, span[0], span[1]) is False


@pytest.mark.parametrize(
    "text,phrase",
    [
        ("I had no chest pain yesterday but today it started", "chest pain"),
        ("I didn't have fever yesterday but now I do", "fever"),
        ("There was no cough earlier but I have cough now", "cough"),
        ("I had no headache earlier but it started this evening", "headache"),
        ("The pain was gone yesterday but it came back today", "pain"),
    ],
)
def test_temporal_reassertion_overrides_historical_negation(text, phrase):
    span = find_phrase_span(text, phrase)
    assert span is not None
    assert temporal_reassertion(text, phrase) is True
    assert is_negated(text, span[0], span[1]) is False


@pytest.mark.parametrize(
    "text",
    [
        "Cough cannot be ruled out",
        "Cough can not be ruled out",
        "Cough cannot rule out",
        "Cough not ruled out",
        "Cannot rule out cough",
        "Not sure if I have cough",
        "Not only cough, I also have fever",
    ],
)
def test_supported_pseudo_negation_phrases_remain_positive(text):
    span = find_phrase_span(text, "cough")
    assert span is not None
    assert is_negated(text, span[0], span[1]) is False


@pytest.mark.parametrize(
    "text",
    [
        "Cough ruled out",
        "Cough was ruled out",
        "No cough",
        "I don't have cough",
        "Cough resolved",
    ],
)
def test_true_negation_remains_negation(text):
    span = find_phrase_span(text, "cough")
    assert span is not None
    assert is_negated(text, span[0], span[1]) is True


def test_historical_negative_without_reassertion_stays_negated():
    text = "I had no fever yesterday"
    span = find_phrase_span(text, "fever")
    assert span is not None
    assert temporal_reassertion(text, "fever") is False
    assert is_negated(text, span[0], span[1]) is True


@pytest.mark.parametrize(
    "text,phrase",
    [
        ("I had fever yesterday but I don't have it now", "fever"),
        ("The cough started yesterday but it has resolved", "cough"),
    ],
)
def test_temporal_resolution_overrides_historical_positive(text, phrase):
    span = find_phrase_span(text, phrase)
    assert span is not None
    assert temporal_resolution(text, phrase) is True
    assert is_negated(text, span[0], span[1]) is True
