import json

from evaluation.evidence_extraction_benchmark import (
    CASE_PATH,
    load_cases,
    run_cases,
    validate_cases,
)


def test_benchmark_cases_are_unique_and_use_known_evidence_codes():
    cases = load_cases(CASE_PATH)
    validate_cases(cases)
    assert len(cases) == len({case["id"] for case in cases})


def test_benchmark_does_not_include_diagnostic_leakage():
    cases = load_cases(CASE_PATH)
    assert all("PATHOLOGY" not in case["input_text"] for case in cases)
    assert all("DIFFERENTIAL_DIAGNOSIS" not in case["input_text"] for case in cases)


def test_unsupported_cases_can_explicitly_expect_no_evidence():
    cases = load_cases(CASE_PATH)
    unsupported = [case for case in cases if case["category"].startswith(("V_", "X_"))]
    assert unsupported
    assert all(case["expected_positive_evidence"] == [] for case in unsupported if case["category"].startswith("V_"))


def test_benchmark_execution_is_deterministic():
    cases = load_cases(CASE_PATH)
    first = run_cases(cases)
    second = run_cases(json.loads(json.dumps(cases)))
    assert first == second
