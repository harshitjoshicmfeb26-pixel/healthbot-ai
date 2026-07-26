"""
tests/test_predictor_explain.py
─────────────────────────────────
Tests for model.predictor.explain_case() — the dependency-free,
coefficient-based explainability added when the Gradio UI was replaced
with the Flask API (see docs/CHANGELOG.md). These run against the real
trained artifacts in saved_models/, same as the rest of the predictor.
"""

from model.predictor import explain_case, predict_case


def test_explain_case_returns_contributions_for_a_real_case():
    codes = ["E_66", "E_201", "E_214"]  # shortness of breath, cough, wheeze
    preds = predict_case(age=55, sex="M", evidences=codes, initial_evidence="E_66", top_n=1)
    assert preds, "Sanity check: the model should return at least one prediction."

    result = explain_case(age=55, sex="M", evidences=codes, initial_evidence="E_66")

    assert result["pathology"] == preds[0]["pathology"]
    assert result["note"] == ""
    assert result["contributions"], "A real case with matched evidence should yield contributions."
    for row in result["contributions"]:
        assert {"feature", "meaning", "weight"} <= row.keys()
        assert row["weight"] > 0
        assert "Unknown evidence" not in row["meaning"], (
            f"Token {row['feature']!r} did not decode to a readable phrase."
        )


def test_explain_case_no_bigram_or_redundant_tokens_leak_into_output():
    codes = ["E_66", "E_201", "E_214"]
    result = explain_case(age=55, sex="M", evidences=codes, initial_evidence="E_66")
    for row in result["contributions"]:
        assert " " not in row["feature"], "Bigram tokens cannot be cleanly humanized; should be filtered."
        assert not row["feature"].startswith("initial_in_evidence_"), (
            "The redundant initial-in-evidence flag token should not appear in the explanation."
        )


def test_explain_case_handles_empty_evidence_gracefully():
    result = explain_case(age=30, sex="F", evidences=[], initial_evidence="")
    assert result["contributions"] == []
    assert result["note"]


def test_explain_case_can_target_a_specific_pathology():
    codes = ["E_66", "E_201", "E_214"]
    preds = predict_case(age=55, sex="M", evidences=codes, initial_evidence="E_66", top_n=5)
    target = preds[-1]["pathology"]  # a lower-ranked candidate, not the top one

    result = explain_case(age=55, sex="M", evidences=codes, initial_evidence="E_66", pathology=target)
    assert result["pathology"] == target
