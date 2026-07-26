"""
tests/test_severity_engine.py
────────────────────────────────
Tests for utils.severity_engine against the bundled demo condition metadata.
"""

from utils.severity_engine import describe_severity_scale, evaluate_differential_severity


def test_describe_severity_scale_runs_and_mentions_known_conditions():
    output = describe_severity_scale()
    assert "Anaphylaxis" in output
    assert "URTI" in output


def test_high_acuity_candidate_detected_even_at_low_rank():
    predictions = [
        {"pathology": "Allergic sinusitis", "disease": "Allergic sinusitis", "confidence": 0.55},
        {"pathology": "Possible NSTEMI / STEMI", "disease": "Possible NSTEMI / STEMI", "confidence": 0.20},
        {"pathology": "URTI", "disease": "URTI", "confidence": 0.15},
    ]
    result = evaluate_differential_severity(predictions)
    assert result.any_high_acuity_candidate is True
    high_acuity_names = {f.pathology for f in result.findings if f.is_high_acuity}
    assert "Possible NSTEMI / STEMI" in high_acuity_names


def test_no_high_acuity_candidate_when_all_mild():
    predictions = [
        {"pathology": "URTI", "disease": "URTI", "confidence": 0.6},
        {"pathology": "Acute otitis media", "disease": "Acute otitis media", "confidence": 0.3},
    ]
    result = evaluate_differential_severity(predictions)
    assert result.any_high_acuity_candidate is False


def test_unknown_pathology_degrades_gracefully():
    predictions = [{"pathology": "Not A Real Disease", "disease": "Not A Real Disease", "confidence": 0.9}]
    result = evaluate_differential_severity(predictions)
    assert result.any_high_acuity_candidate is False
    assert result.findings[0].severity is None


def test_empty_predictions():
    result = evaluate_differential_severity([])
    assert result.findings == []
    assert result.any_high_acuity_candidate is False
