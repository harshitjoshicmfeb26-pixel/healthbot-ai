"""Regression tests for separated core and similar-case artifact loading."""

from types import SimpleNamespace

import pytest

import model.predictor as predictor


def _reset_predictor_state() -> None:
    predictor._vectorizer = None
    predictor._clf = None
    predictor._le = None
    predictor._X_search = None
    predictor._search_cases = None
    predictor._metadata = None


@pytest.fixture(autouse=True)
def reset_predictor_state():
    _reset_predictor_state()
    yield
    _reset_predictor_state()


def test_all_artifacts_support_prediction_explanation_and_search():
    codes = ["E_66", "E_201", "E_214"]

    predictions = predictor.predict_case(55, "M", codes, "E_66", top_n=1)
    explanation = predictor.explain_case(55, "M", codes, "E_66")
    similar_cases = predictor.semantic_search_case(55, "M", codes, "E_66", top_k=1)

    assert predictions
    assert explanation["pathology"] == predictions[0]["pathology"]
    if predictor.EMBEDDINGS_PATH.exists() and predictor.SEARCH_CASES_PATH.exists():
        assert similar_cases
    else:
        assert similar_cases == []


def test_missing_search_artifacts_do_not_block_core_or_trigger_retraining(monkeypatch, tmp_path):
    codes = ["E_66", "E_201", "E_214"]
    predictor._ensure_core_loaded()
    monkeypatch.setattr(predictor, "EMBEDDINGS_PATH", tmp_path / "missing_matrix.pkl")
    monkeypatch.setattr(predictor, "SEARCH_CASES_PATH", tmp_path / "missing_cases.pkl")

    retraining_calls = []

    def fail_if_retraining(*args, **kwargs):
        retraining_calls.append((args, kwargs))
        raise AssertionError("search artifact absence triggered retraining")

    monkeypatch.setattr(predictor, "subprocess", SimpleNamespace(run=fail_if_retraining))

    predictions = predictor.predict_case(55, "M", codes, "E_66", top_n=1)
    explanation = predictor.explain_case(55, "M", codes, "E_66")
    similar_cases = predictor.semantic_search_case(55, "M", codes, "E_66", top_k=1)

    assert predictions
    assert explanation["pathology"] == predictions[0]["pathology"]
    assert similar_cases == []
    assert retraining_calls == []


def test_missing_core_artifact_preserves_core_retraining_behavior(monkeypatch, tmp_path):
    predictor._ensure_core_loaded()
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "missing_classifier.pkl")
    predictor._clf = None

    retraining_calls = []

    def record_retraining(*args, **kwargs):
        retraining_calls.append((args, kwargs))
        raise RuntimeError("core retraining requested")

    monkeypatch.setattr(predictor, "subprocess", SimpleNamespace(run=record_retraining))

    with pytest.raises(RuntimeError, match="core retraining requested"):
        predictor.predict_case(55, "M", ["E_66"], "E_66", top_n=1)

    assert retraining_calls
