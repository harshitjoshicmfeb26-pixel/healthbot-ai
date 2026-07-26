"""
tests/test_negation.py
───────────────────────
Unit tests for utils.negation — the NegEx-style negation detector added in
the refactor (see docs/CHANGELOG_REFACTOR.md).
"""

import pytest

from utils.negation import filter_negated, find_phrase_span, is_negated


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
