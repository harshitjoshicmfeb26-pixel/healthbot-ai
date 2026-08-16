"""Offline, non-production comparison of the two saved HealthBot models.

This module deliberately loads artifacts directly instead of importing the
production predictor loader.  It therefore cannot trigger automatic model
training while evaluating an existing artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from config import (
    ENCODER_PATH,
    MODEL_PATH,
    SIMPLIFIED_DISEASE_MODEL_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VECTORIZER_PATH,
)
from model.predict_disease import build_model_input
from utils.clinical_case_features import build_case_feature_text, evidence_codes
from utils.ddxplus_decoder import decode_evidence, evidence_metadata, split_evidence_code


RECORD_COLUMNS = (
    "AGE",
    "SEX",
    "EVIDENCES",
    "INITIAL_EVIDENCE",
    "PATHOLOGY",
    "DIFFERENTIAL_DIAGNOSIS",
)
FEATURE_COLUMNS = ("AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE")
CSV_COLUMNS = list(RECORD_COLUMNS)
FAMILY_RE = re.compile(r"\b(family|genetic|mother|father|sister|brother|parent|relative|relatives)\b", re.I)
LOCATION_BASE_CODES = {"E_55", "E_57"}
SEVERITY_BASE_CODES = {"E_56", "E_134"}
UNKNOWN = "unknown"
EVIDENCE_CODE_RE = re.compile(r"E_\d+(?:_@_(?:V_)?\d+)?", re.I)


def _clean_scalar(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def _parse_literal(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return value
    text = _clean_scalar(value)
    if not text:
        return []
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _canonical_sequence(value: Any) -> Any:
    parsed = _parse_literal(value)
    if isinstance(parsed, dict):
        return {str(k): _canonical_sequence(v) for k, v in sorted(parsed.items(), key=lambda item: str(item[0]))}
    if isinstance(parsed, (list, tuple)):
        return [_canonical_sequence(item) for item in parsed]
    return _clean_scalar(parsed)


def _canonical_evidence(value: Any) -> list[str]:
    # Hash normalization is intentionally lightweight: the evaluator must
    # process more than one million train rows without reparsing the large
    # differential-diagnosis payload on every row.  The actual model feature
    # transformation still uses the validated production parser below.
    text = _clean_scalar(value).upper()
    return sorted(set(EVIDENCE_CODE_RE.findall(text)))


def _canonical_differential(value: Any) -> Any:
    return _clean_scalar(value)


def canonical_identity(row: dict[str, Any], *, include_diagnostic: bool = True) -> str:
    """Return a stable identity for an exact record or patient feature set."""
    payload: dict[str, Any] = {
        "AGE": _clean_scalar(row.get("AGE")),
        "SEX": _clean_scalar(row.get("SEX")).upper(),
        "EVIDENCES": _canonical_evidence(row.get("EVIDENCES")),
        "INITIAL_EVIDENCE": _clean_scalar(row.get("INITIAL_EVIDENCE")).upper(),
    }
    if include_diagnostic:
        payload["PATHOLOGY"] = _clean_scalar(row.get("PATHOLOGY"))
        payload["DIFFERENTIAL_DIAGNOSIS"] = _canonical_differential(row.get("DIFFERENTIAL_DIAGNOSIS"))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _metadata_entry(code: str) -> dict[str, Any]:
    base, _ = split_evidence_code(code)
    return evidence_metadata().get(base, {})


def _decoded_meaning(code: str) -> str:
    return str(decode_evidence(code).get("meaning") or code).strip()


def _row_codes(row: dict[str, Any]) -> list[str]:
    return sorted(set(evidence_codes(row.get("EVIDENCES"))))


def _is_family_code(code: str) -> bool:
    question = str(_metadata_entry(code).get("question_en", ""))
    return bool(FAMILY_RE.search(question))


def _antecedent_codes(row: dict[str, Any], family: bool) -> list[str]:
    selected: list[str] = []
    for code in _row_codes(row):
        entry = _metadata_entry(code)
        if not entry.get("is_antecedent"):
            continue
        if _is_family_code(code) != family:
            continue
        selected.append(code)
    return selected


def _render_codes(codes: Iterable[str]) -> str:
    meanings = [_decoded_meaning(code) for code in sorted(set(codes))]
    return "; ".join(meanings) if meanings else UNKNOWN


def _pain_location(row: dict[str, Any]) -> str:
    locations: list[str] = []
    for code in _row_codes(row):
        base, value = split_evidence_code(code)
        if base not in LOCATION_BASE_CODES or not value:
            continue
        meaning = _decoded_meaning(code)
        if meaning.lower().endswith(" nowhere") or meaning.lower() == "nowhere":
            continue
        locations.append(meaning)
    return "; ".join(sorted(set(locations))) if locations else UNKNOWN


def _pain_severity(row: dict[str, Any]) -> str:
    values: list[str] = []
    for code in _row_codes(row):
        base, value = split_evidence_code(code)
        if base not in SEVERITY_BASE_CODES or not value:
            continue
        decoded = decode_evidence(code)
        rendered = decoded.get("value") or value
        values.append(f"pain intensity: {rendered}")
    return "; ".join(sorted(set(values))) if values else UNKNOWN


def reconstruct_simplified_fields(row: dict[str, Any]) -> dict[str, str]:
    """Build the fixed simplified-model contract without diagnostic leakage."""
    codes = _row_codes(row)
    initial = _clean_scalar(row.get("INITIAL_EVIDENCE")).upper()
    symptom_codes = list(codes)
    if initial and initial not in symptom_codes:
        symptom_codes.append(initial)
    symptom_codes = sorted(set(symptom_codes))
    return {
        "age": _clean_scalar(row.get("AGE")) or UNKNOWN,
        "gender": {"F": "female", "M": "male"}.get(_clean_scalar(row.get("SEX")).upper(), UNKNOWN),
        "symptoms_text": _render_codes(symptom_codes),
        "pain_location": _pain_location(row),
        "previous_disease_or_history": _render_codes(_antecedent_codes(row, family=False)),
        "genetic_or_family_history": _render_codes(_antecedent_codes(row, family=True)),
        "duration": UNKNOWN,
        "severity": _pain_severity(row),
    }


def reconstructed_model_text(row: dict[str, Any]) -> str:
    fields = reconstruct_simplified_fields(row)
    return build_model_input(**fields)


def _row_dicts(frame: pd.DataFrame) -> Iterable[dict[str, Any]]:
    for row in frame[CSV_COLUMNS].to_dict(orient="records"):
        yield row


def load_evaluation_population(chunk_size: int = 50_000) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load test rows while excluding normalized exact train-record duplicates."""
    train_exact: set[str] = set()
    train_features: set[str] = set()
    for chunk in pd.read_csv(TRAIN_DATA_PATH, usecols=CSV_COLUMNS, dtype=str, chunksize=chunk_size):
        for row in _row_dicts(chunk):
            train_exact.add(canonical_identity(row, include_diagnostic=True))
            train_features.add(canonical_identity(row, include_diagnostic=False))

    kept: list[pd.DataFrame] = []
    excluded = 0
    remaining_feature_overlap = 0
    original = 0
    for chunk in pd.read_csv(TEST_DATA_PATH, usecols=CSV_COLUMNS, dtype=str, chunksize=chunk_size):
        original += len(chunk)
        keep_mask: list[bool] = []
        for row in _row_dicts(chunk):
            exact = canonical_identity(row, include_diagnostic=True)
            feature = canonical_identity(row, include_diagnostic=False)
            is_excluded = exact in train_exact
            keep_mask.append(not is_excluded)
            if is_excluded:
                excluded += 1
            elif feature in train_features:
                remaining_feature_overlap += 1
        if any(keep_mask):
            kept.append(chunk.loc[keep_mask].copy())

    population = pd.concat(kept, ignore_index=True)
    stats = {
        "original_test_rows": original,
        "excluded_exact_train_duplicates": excluded,
        "remaining_rows": len(population),
        "remaining_feature_level_train_overlap": remaining_feature_overlap,
        "train_exact_identity_count": len(train_exact),
        "train_feature_identity_count": len(train_features),
    }
    return population, stats


def _structured_probabilities(texts: list[str], vectorizer: Any, classifier: Any) -> np.ndarray:
    return np.asarray(classifier.predict_proba(vectorizer.transform(texts)), dtype=float)


def _probabilities(model: Any, texts: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts), dtype=float)
    scores = np.asarray(model.decision_function(texts), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _align_probabilities(probabilities: np.ndarray, classes: Iterable[Any], class_names: list[str]) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(class_names)), dtype=float)
    positions = {name: index for index, name in enumerate(class_names)}
    for index, label in enumerate(classes):
        name = str(label)
        if name in positions:
            aligned[:, positions[name]] = probabilities[:, index]
    return aligned


def _top_names(probabilities: np.ndarray, class_names: list[str], n: int) -> list[list[str]]:
    indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :n]
    return [[class_names[index] for index in row] for row in indices]


def _ece(probabilities: np.ndarray, y_true: np.ndarray, class_names: list[str], bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(class_names)[probabilities.argmax(axis=1)]
    correct = predictions == y_true
    total = len(y_true)
    result = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            result += mask.mean() * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(result)


def _brier(probabilities: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> float:
    positions = {name: index for index, name in enumerate(class_names)}
    target = np.zeros_like(probabilities)
    for row, label in enumerate(y_true):
        if label in positions:
            target[row, positions[label]] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _metric_rows(name: str, probabilities: np.ndarray, y_true: np.ndarray, class_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    top = _top_names(probabilities, class_names, 5)
    top1 = np.asarray([row[0] for row in top])
    top3 = [set(row[:3]) for row in top]
    top5 = [set(row[:5]) for row in top]
    max_conf = probabilities.max(axis=1)
    correct = top1 == y_true
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, top1, labels=class_names, zero_division=0
    )
    summary = {
        "model": name,
        "rows": int(len(y_true)),
        "top1_accuracy": float(accuracy_score(y_true, top1)),
        "macro_f1": float(f1_score(y_true, top1, labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, top1, labels=class_names, average="weighted", zero_division=0)),
        "top3_accuracy": float(np.mean([label in candidates for label, candidates in zip(y_true, top3)])),
        "top5_accuracy": float(np.mean([label in candidates for label, candidates in zip(y_true, top5)])),
        "mean_max_confidence": float(max_conf.mean()),
        "median_max_confidence": float(np.median(max_conf)),
        "confidence_quantiles": {str(q): float(np.quantile(max_conf, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "mean_confidence_correct": float(max_conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(max_conf[~correct].mean()) if (~correct).any() else None,
        "ece_10_bins": _ece(probabilities, y_true, class_names),
        "multiclass_brier": _brier(probabilities, y_true, class_names),
    }
    per_class = pd.DataFrame({
        "model": name,
        "class": class_names,
        "recall": recall,
        "f1": f1,
        "support": support,
        "precision": precision,
    })
    return summary, per_class


def _latency(texts: list[str], vectorizer: Any, structured: Any, simplified: Any, sample_size: int = 2000) -> dict[str, Any]:
    sample = texts[: min(sample_size, len(texts))]

    def timed(fn: Any) -> dict[str, float]:
        values = []
        for text in sample:
            started = time.perf_counter()
            fn(text)
            values.append((time.perf_counter() - started) * 1000.0)
        return {
            "sample_rows": len(values),
            "median_ms_per_row": float(np.median(values)),
            "p95_ms_per_row": float(np.quantile(values, 0.95)),
        }

    return {
        "latency_sample_policy": "first 2,000 rows after exact-duplicate exclusion; models loaded once",
        "structured_warm_inference": timed(lambda text: structured.predict_proba(vectorizer.transform([text]))),
        "simplified_warm_inference": timed(lambda text: _probabilities(simplified, [text])),
    }


def run(output_dir: Path, batch_size: int = 4096) -> dict[str, Any]:
    started = time.perf_counter()
    population, population_stats = load_evaluation_population()
    y_true = population["PATHOLOGY"].astype(str).to_numpy()
    rows = list(_row_dicts(population))

    cold_started = time.perf_counter()
    vectorizer = joblib.load(VECTORIZER_PATH)
    structured = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    simplified = joblib.load(SIMPLIFIED_DISEASE_MODEL_PATH)
    cold_seconds = time.perf_counter() - cold_started

    structured_classes = [str(encoder.inverse_transform([label])[0]) for label in structured.classes_]
    simplified_classes = [str(label) for label in simplified.classes_]
    class_names = sorted(set(structured_classes) | set(simplified_classes) | set(y_true))

    structured_parts: list[np.ndarray] = []
    simplified_parts: list[np.ndarray] = []
    structured_texts: list[str] = []
    simplified_texts: list[str] = []
    mapping_counts = {key: 0 for key in ("pain_location", "previous_disease_or_history", "genetic_or_family_history", "severity")}
    severity_values: dict[str, int] = {}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        native_texts = [build_case_feature_text(row["AGE"], row["SEX"], row["EVIDENCES"], row["INITIAL_EVIDENCE"]) for row in batch]
        reconstructed = [reconstruct_simplified_fields(row) for row in batch]
        simple_texts = [build_model_input(**fields) for fields in reconstructed]
        structured_texts.extend(native_texts)
        simplified_texts.extend(simple_texts)
        for fields in reconstructed:
            for key in mapping_counts:
                if fields[key] != UNKNOWN:
                    mapping_counts[key] += 1
            if fields["severity"] != UNKNOWN:
                severity_values[fields["severity"]] = severity_values.get(fields["severity"], 0) + 1
        structured_parts.append(_align_probabilities(_structured_probabilities(native_texts, vectorizer, structured), structured_classes, class_names))
        simplified_parts.append(_align_probabilities(_probabilities(simplified, simple_texts), simplified_classes, class_names))

    structured_probabilities = np.vstack(structured_parts)
    simplified_probabilities = np.vstack(simplified_parts)
    structured_summary, structured_per_class = _metric_rows("structured_native_ddxplus", structured_probabilities, y_true, class_names)
    simplified_summary, simplified_per_class = _metric_rows("simplified_reconstructed_ddxplus", simplified_probabilities, y_true, class_names)

    structured_top1 = np.asarray(_top_names(structured_probabilities, class_names, 1), dtype=object)[:, 0]
    simplified_top1 = np.asarray(_top_names(simplified_probabilities, class_names, 1), dtype=object)[:, 0]
    structured_top3 = _top_names(structured_probabilities, class_names, 3)
    simplified_top3 = _top_names(simplified_probabilities, class_names, 3)
    jaccard = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(structured_top3, simplified_top3)]
    agreement = {
        "top1_agreement_rate": float(np.mean(structured_top1 == simplified_top1)),
        "top3_overlap_rate": float(np.mean([bool(set(a) & set(b)) for a, b in zip(structured_top3, simplified_top3)])),
        "mean_top3_jaccard_overlap": float(np.mean(jaccard)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([structured_summary, simplified_summary])
    per_class = structured_per_class.drop(columns="model").merge(
        simplified_per_class.drop(columns="model"),
        on="class",
        suffixes=("_structured", "_simplified"),
    )
    metrics.to_csv(output_dir / "model_comparison_metrics.csv", index=False)
    per_class.to_csv(output_dir / "per_class_comparison.csv", index=False)

    latency = _latency(structured_texts, vectorizer, structured, simplified)
    summary = {
        "protocol": {
            "evaluation_population": "official test.csv minus normalized exact train-record duplicates",
            "structured_input": "native DDXPlus evidence feature text",
            "simplified_input": "reconstructed DDXPlus-derived text and slots",
            "clinical_superiority_claim": False,
        },
        "population": population_stats,
        "mapping_coverage": {
            **mapping_counts,
            "total_rows": len(rows),
            "coverage_fraction": {key: mapping_counts[key] / len(rows) for key in mapping_counts},
            "severity_values": severity_values,
            "duration": {"unknown_rows": len(rows), "non_unknown_rows": 0},
        },
        "cold_load": {"combined_artifact_load_seconds": cold_seconds},
        "models": [structured_summary, simplified_summary],
        "agreement": agreement,
        "latency": latency,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "model_comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "agreement_summary.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    (output_dir / "calibration_summary.json").write_text(json.dumps({"structured": structured_summary, "simplified": simplified_summary}, indent=2), encoding="utf-8")
    (output_dir / "latency_summary.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")

    print("STRUCTURED_INPUT=native DDXPlus feature representation")
    print("SIMPLIFIED_INPUT=reconstructed DDXPlus-derived text/slots")
    print("SYMMETRIC_NATIVE_COMPARISON=False")
    print("CLINICAL_SUPERIORITY_CLAIM=False")
    print(json.dumps({"population": population_stats, "models": metrics.to_dict(orient="records"), "agreement": agreement}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    run(args.output_dir, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
