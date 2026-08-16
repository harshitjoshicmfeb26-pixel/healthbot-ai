"""Offline benchmark for the current natural-language evidence extractor.

This module intentionally calls the existing production utilities without
changing them. Ollama NLU and BioBERT are disabled before imports so results
are deterministic and offline.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["OLLAMA_NLU_ENABLED"] = "False"
os.environ["BIOBERT_ENABLED"] = "False"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ddxplus_decoder import (  # noqa: E402
    evidence_metadata,
    infer_evidence_codes_from_text,
    select_initial_evidence,
    split_evidence_code,
)
from utils.language_detector import detect_language  # noqa: E402
from utils.multilingual_normalizer import normalize_symptoms  # noqa: E402
from utils.red_flag_rules import detect_red_flags  # noqa: E402


CASE_PATH = Path(__file__).with_name("evidence_benchmark_cases.json")
RESULTS_DIR = Path(__file__).with_name("results")

# Expected INITIAL_EVIDENCE follows official DDXPlus semantics: categorical
# pain details remain in EVIDENCES, while the presenting complaint uses its
# base code. Antecedent-only cases intentionally have no initial evidence.


def load_cases(path: Path = CASE_PATH) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark fixture must contain a non-empty list.")
    ids = [str(case.get("id", "")) for case in cases]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Benchmark IDs must be non-empty and unique.")
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    metadata = evidence_metadata()
    required = {"id", "category", "language", "input_text", "expected_positive_evidence", "expected_denied_evidence", "notes"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"{case.get('id')}: missing fields {sorted(missing)}")
        for field in ("expected_positive_evidence", "expected_denied_evidence"):
            for code in case[field]:
                base, _ = split_evidence_code(code)
                if base not in metadata:
                    raise ValueError(f"{case['id']}: unknown DDXPlus evidence code {code}")
        expected_initial = str(case.get("expected_initial_evidence", ""))
        if expected_initial:
            base, _ = split_evidence_code(expected_initial)
            if base not in metadata:
                raise ValueError(f"{case['id']}: unknown initial evidence code {expected_initial}")
            if "_@_" in expected_initial:
                raise ValueError(f"{case['id']}: INITIAL_EVIDENCE must use a base code")
            if metadata[base].get("is_antecedent"):
                raise ValueError(f"{case['id']}: antecedent evidence cannot be INITIAL_EVIDENCE")


def _error_classes(case: dict[str, Any], actual_positive: set[str], actual_denied: set[str], red_flag_match: bool) -> list[str]:
    expected_positive = set(case["expected_positive_evidence"])
    expected_denied = set(case["expected_denied_evidence"])
    errors: list[str] = []
    if expected_positive - actual_positive:
        category = case["category"]
        if category.startswith("C_"):
            errors.append("typo failure")
        elif category.startswith("G_"):
            errors.append("normalization-to-evidence gap")
            errors.append("code-switching failure")
        elif category.startswith("J_"):
            errors.append("negation trigger gap")
        elif category.startswith("M_"):
            errors.append("temporal reasoning error")
        elif category.startswith("R_") or category.startswith("Q_"):
            errors.append("categorical mapping error")
        elif category.startswith("X_") or category.startswith("V_"):
            errors.append("unsupported concept")
        else:
            errors.append("lexical coverage gap")
    if expected_denied - actual_denied:
        category = case["category"]
        if category.startswith("J_"):
            errors.append("negation trigger gap")
        elif category.startswith("M_"):
            errors.append("temporal reasoning error")
        else:
            errors.append("negation-scope error")
    if actual_positive - expected_positive:
        if case["category"].startswith("U_") or case["category"].startswith("T_"):
            errors.append("nested alias false positive")
        elif case["category"].startswith("X_") or case["category"].startswith("V_"):
            errors.append("ambiguity")
        else:
            errors.append("fuzzy false positive" if any("fuzzy" in str(item) for item in []) else "lexical coverage gap")
    if not red_flag_match:
        errors.append("red-flag detection failure")
    return list(dict.fromkeys(errors))


def run_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_cases(cases)
    records = []
    for case in cases:
        text = case["input_text"]
        normalized = normalize_symptoms(text)
        inferred = infer_evidence_codes_from_text(text, include_denied=True)
        positive = sorted({item["code"] for item in inferred if not item.get("negated")})
        denied = sorted({item["code"] for item in inferred if item.get("negated")})
        positive_matches = [item for item in inferred if not item.get("negated")]
        actual_initial = select_initial_evidence(positive_matches) if positive_matches else ""
        red_flag = detect_red_flags(text)
        expected_red = case.get("expected_red_flag")
        errors = _error_classes(case, set(positive), set(denied), red_flag["has_red_flag"] == expected_red) if expected_red is not None else []
        records.append({
            "id": case["id"],
            "category": case["category"],
            "language": case["language"],
            "input_text": text,
            "detected_language": normalized["detected_language"].get("language", "unknown"),
            "normalized_text": normalized["normalized_text"],
            "actual_positive_evidence": sorted(positive),
            "actual_denied_evidence": sorted(denied),
            "expected_positive_evidence": sorted(case["expected_positive_evidence"]),
            "expected_denied_evidence": sorted(case["expected_denied_evidence"]),
            "expected_initial_evidence": case.get("expected_initial_evidence", ""),
            "actual_initial_evidence": actual_initial,
            "expected_red_flag": expected_red,
            "actual_red_flag": red_flag["has_red_flag"],
            "matched_red_flags": "; ".join(red_flag["matched_flags"]),
            "error_classes": "; ".join(errors),
            "notes": case["notes"],
        })
    return records


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _evidence_metrics(records: list[dict[str, Any]], expected_key: str, actual_key: str) -> dict[str, Any]:
    tp = fp = fn = exact = 0
    false_positive_counts = []
    missed_cases = 0
    for record in records:
        expected = set(record[expected_key])
        actual = set(record[actual_key])
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
        false_positive_counts.append(len(actual - expected))
        missed_cases += bool(expected - actual)
        exact += expected == actual
    return {
        **_prf(tp, fp, fn),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "exact_set_match_rate": exact / len(records),
        "average_false_positive_count": sum(false_positive_counts) / len(records),
        "utterances_with_false_positive_pct": 100 * sum(value > 0 for value in false_positive_counts) / len(records),
        "utterances_with_missed_evidence_pct": 100 * missed_cases / len(records),
    }


def _group_metrics(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record[group_key], []).append(record)
    rows = []
    for group, items in sorted(grouped.items()):
        metrics = _evidence_metrics(items, "expected_positive_evidence", "actual_positive_evidence")
        rows.append({"group": group, "rows": len(items), **metrics})
    return rows


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    positive = _evidence_metrics(records, "expected_positive_evidence", "actual_positive_evidence")
    denied = _evidence_metrics(records, "expected_denied_evidence", "actual_denied_evidence")
    initial_cases = [record for record in records if record["expected_initial_evidence"]]
    initial_correct = sum(record["expected_initial_evidence"] == record["actual_initial_evidence"] for record in initial_cases)
    red_cases = [record for record in records if record["expected_red_flag"] is not None]
    red_tp = sum(record["expected_red_flag"] and record["actual_red_flag"] for record in red_cases)
    red_fp = sum(not record["expected_red_flag"] and record["actual_red_flag"] for record in red_cases)
    red_fn = sum(record["expected_red_flag"] and not record["actual_red_flag"] for record in red_cases)
    error_counts = Counter(
        error
        for record in records
        for error in record["error_classes"].split("; ")
        if error
    )
    return {
        "benchmark": {
            "case_count": len(records),
            "offline": True,
            "ollama_enabled": False,
            "biobert_enabled": False,
            "production_code_modified": False,
        },
        "positive_evidence": positive,
        "denied_evidence": denied,
        "initial_evidence": {
            "defined_case_count": len(initial_cases),
            "correct_count": initial_correct,
            "accuracy": initial_correct / len(initial_cases) if initial_cases else None,
        },
        "red_flags": {
            "case_count": len(red_cases),
            "false_negative_count": red_fn,
            **_prf(red_tp, red_fp, red_fn),
        },
        "language_metrics": _group_metrics(records, "language"),
        "category_metrics": _group_metrics(records, "category"),
        "failure_class_counts": dict(error_counts.most_common()),
    }


def write_results(records: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path = RESULTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence_extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "evidence_extraction_failures.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record["error_classes"]:
                row = dict(record)
                for key, value in row.items():
                    if isinstance(value, list):
                        row[key] = "; ".join(value)
                writer.writerow(row)

    metric_rows = []
    for name, metrics in (("positive", summary["positive_evidence"]), ("denied", summary["denied_evidence"])):
        metric_rows.append({"scope": name, **metrics})
    with (output_dir / "evidence_extraction_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in metric_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    for filename, key in (("evidence_extraction_language_metrics.csv", "language_metrics"), ("evidence_extraction_category_metrics.csv", "category_metrics")):
        rows = summary[key]
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            fields = sorted({field for row in rows for field in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def run(cases_path: Path = CASE_PATH, output_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    cases = load_cases(cases_path)
    records = run_cases(cases)
    summary = build_summary(records)
    write_results(records, summary, output_dir)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
