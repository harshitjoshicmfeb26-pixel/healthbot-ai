"""
tests/test_clinical_case_features.py
───────────────────────────────────────
Tests for utils.clinical_case_features — the structured-record-to-token
pipeline that both training (model/train.py) and inference
(model/predictor.py) rely on.
"""

from utils.clinical_case_features import (
    age_bucket,
    build_case_feature_text,
    clean_sex,
    decade_bucket,
    evidence_codes,
)


def test_age_bucket_boundaries():
    assert age_bucket(0) == "age_infant"
    assert age_bucket(10) == "age_child"
    assert age_bucket(16) == "age_teen"
    assert age_bucket(30) == "age_adult_18_39"
    assert age_bucket(50) == "age_adult_40_64"
    assert age_bucket(70) == "age_senior"
    assert age_bucket("not a number") == "age_unknown"


def test_decade_bucket():
    assert decade_bucket(34) == "age_decade_30"
    assert decade_bucket(None) == "age_decade_unknown"


def test_clean_sex_normalizes_variants():
    assert clean_sex("female") == "F"
    assert clean_sex("M") == "M"
    assert clean_sex("") == "U"
    assert clean_sex(None) == "U"


def test_evidence_codes_extracts_valid_codes_only():
    codes = evidence_codes(["E_53", "E_55_@_V_29", "not_a_code", "E_91"])
    assert "E_53" in codes
    assert "E_55_@_V_29" in codes
    assert "E_91" in codes
    assert "not_a_code" not in codes


def test_build_case_feature_text_contains_expected_tokens():
    text = build_case_feature_text(
        age=34, sex="F", evidences=["E_53", "E_91"], initial_evidence="E_53"
    )
    tokens = text.split()
    assert "age_adult_18_39" in tokens
    assert "age_decade_30" in tokens
    assert "sex_F" in tokens
    assert "ev_E_53" in tokens
    assert "ev_E_91" in tokens


def test_build_case_feature_text_handles_missing_fields():
    # Should not raise even with nothing provided.
    text = build_case_feature_text()
    assert "sex_U" in text.split()
