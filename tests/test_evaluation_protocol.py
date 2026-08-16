from evaluation.compare_models import (
    canonical_identity,
    reconstruct_simplified_fields,
    reconstructed_model_text,
)


def _row():
    return {
        "AGE": "42",
        "SEX": "F",
        "EVIDENCES": "['E_91', 'E_56_@_8', 'E_55_@_V_29', 'E_87']",
        "INITIAL_EVIDENCE": "E_91",
        "PATHOLOGY": "Asthma",
        "DIFFERENTIAL_DIAGNOSIS": "[['Asthma', 0.8]]",
    }


def test_duplicate_identity_is_stable_and_diagnostic_fields_are_separate():
    row = _row()
    assert canonical_identity(row) == canonical_identity(dict(row))
    changed = dict(row, PATHOLOGY="Different")
    assert canonical_identity(row, include_diagnostic=False) == canonical_identity(changed, include_diagnostic=False)
    assert canonical_identity(row) != canonical_identity(changed)


def test_reconstruction_excludes_diagnostic_columns_and_has_fixed_duration():
    fields = reconstruct_simplified_fields(_row())
    text = reconstructed_model_text(_row())
    assert fields["duration"] == "unknown"
    assert "Asthma" not in text
    assert "0.8" not in text
    assert "duration: unknown" in text


def test_severity_uses_only_explicit_pain_intensity():
    row = _row()
    assert reconstruct_simplified_fields(row)["severity"] != "unknown"
    row["EVIDENCES"] = "['E_91']"
    assert reconstruct_simplified_fields(row)["severity"] == "unknown"


def test_mapping_is_deterministic_and_initial_evidence_is_not_duplicated():
    row = _row()
    first = reconstruct_simplified_fields(row)
    second = reconstruct_simplified_fields(row)
    assert first == second
    assert first["symptoms_text"].count("fever") == 1
