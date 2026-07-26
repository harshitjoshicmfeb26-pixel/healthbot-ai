"""
Utilities for the structured clinical evidence dataset.

Expected columns:
AGE, SEX, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE, DIFFERENTIAL_DIAGNOSIS
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable

from utils.ddxplus_decoder import decode_evidence


REQUIRED_COLUMNS = {
    "AGE",
    "SEX",
    "PATHOLOGY",
    "EVIDENCES",
    "INITIAL_EVIDENCE",
    "DIFFERENTIAL_DIAGNOSIS",
}

EVIDENCE_CODE_RE = re.compile(r"E_\d+(?:_@_(?:V_)?\d+)?", re.IGNORECASE)


def parse_list_value(value: Any) -> list:
    """Parse a Python-list-like CSV cell into a list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def evidence_codes(value: Any) -> list[str]:
    """Return normalized evidence codes from a CSV list cell or user text."""
    if isinstance(value, list):
        items = value
    else:
        text = str(value or "")
        parsed = parse_list_value(text)
        if parsed:
            items = parsed
        else:
            items = EVIDENCE_CODE_RE.findall(text)
            if not items:
                items = re.split(r"[\s,;]+", text)

    codes = []
    for item in items:
        code = str(item).strip().strip("'\"")
        if not code:
            continue
        if EVIDENCE_CODE_RE.fullmatch(code):
            codes.append(code.upper())
    return list(dict.fromkeys(codes))


def parse_differential_diagnosis(value: Any, limit: int | None = None) -> list[dict]:
    """Parse DIFFERENTIAL_DIAGNOSIS into [{'condition', 'probability'}]."""
    rows = parse_list_value(value)
    parsed = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            probability = float(item[1])
        except (TypeError, ValueError):
            probability = 0.0
        parsed.append({
            "condition": str(item[0]),
            "probability": probability,
            "probability_pct": f"{probability * 100:.1f}%",
        })
    parsed.sort(key=lambda row: row["probability"], reverse=True)
    return parsed[:limit] if limit else parsed


def age_bucket(age: Any) -> str:
    """Convert age into stable demographic tokens."""
    try:
        age_int = int(float(age))
    except (TypeError, ValueError):
        return "age_unknown"
    if age_int < 1:
        return "age_infant"
    if age_int < 13:
        return "age_child"
    if age_int < 18:
        return "age_teen"
    if age_int < 40:
        return "age_adult_18_39"
    if age_int < 65:
        return "age_adult_40_64"
    return "age_senior"


def decade_bucket(age: Any) -> str:
    try:
        age_int = max(0, min(110, int(float(age))))
    except (TypeError, ValueError):
        return "age_decade_unknown"
    return f"age_decade_{(age_int // 10) * 10}"


def clean_sex(sex: Any) -> str:
    value = str(sex or "").strip().upper()
    if value.startswith("F"):
        return "F"
    if value.startswith("M"):
        return "M"
    return "U"


def _code_token(prefix: str, code: str) -> str:
    token = re.sub(r"[^A-Z0-9]+", "_", str(code).upper()).strip("_")
    return f"{prefix}_{token}" if token else ""


def evidence_feature_tokens(codes: Iterable[str], initial_evidence: str | None = None) -> list[str]:
    tokens = []
    initial = str(initial_evidence or "").strip().upper()
    for code in evidence_codes(list(codes)):
        tokens.append(_code_token("ev", code))
        if "_@_" in code:
            base, value = code.split("_@_", 1)
            tokens.append(_code_token("evbase", base))
            tokens.append(_code_token("evval", value))
        if initial and code == initial:
            tokens.append(_code_token("initial_in_evidence", code))
    return [token for token in tokens if token]


def build_case_feature_text(
    age: Any = None,
    sex: Any = None,
    evidences: Any = None,
    initial_evidence: Any = None,
) -> str:
    """Build the text feature string used by TF-IDF/ML training and inference."""
    sex_value = clean_sex(sex)
    initial = str(initial_evidence or "").strip().upper()
    codes = evidence_codes(evidences)

    tokens = [
        age_bucket(age),
        decade_bucket(age),
        f"sex_{sex_value}",
    ]
    if initial:
        tokens.append(_code_token("initial", initial))
    tokens.extend(evidence_feature_tokens(codes, initial))
    return " ".join(token for token in tokens if token)


def row_to_feature_text(row: Any) -> str:
    return build_case_feature_text(
        age=row["AGE"],
        sex=row["SEX"],
        evidences=row["EVIDENCES"],
        initial_evidence=row["INITIAL_EVIDENCE"],
    )


def row_to_case_record(row: Any) -> dict:
    differentials = parse_differential_diagnosis(row.get("DIFFERENTIAL_DIAGNOSIS", ""), limit=5)
    evidences = evidence_codes(row.get("EVIDENCES", ""))
    return {
        "age": int(row["AGE"]),
        "sex": clean_sex(row["SEX"]),
        "pathology": str(row["PATHOLOGY"]),
        "initial_evidence": str(row["INITIAL_EVIDENCE"]),
        "evidences": evidences,
        "decoded_evidences": [decode_evidence(code) for code in evidences],
        "decoded_initial_evidence": decode_evidence(str(row["INITIAL_EVIDENCE"])),
        "evidence_count": len(evidences),
        "differential_diagnosis": differentials,
        "feature_text": row_to_feature_text(row),
    }


def format_case_record(record: dict, evidence_limit: int = 12, show_codes: bool = False) -> str:
    decoded_rows = record.get("decoded_evidences") or [decode_evidence(code) for code in record.get("evidences", [])]
    if show_codes:
        evidence_preview = ", ".join(
            f"`{row['code']}`={row['meaning']}" for row in decoded_rows[:evidence_limit]
        )
    else:
        evidence_preview = "; ".join(row["meaning"] for row in decoded_rows[:evidence_limit])
    if record.get("evidence_count", 0) > evidence_limit:
        evidence_preview += ", ..."
    diff_preview = ", ".join(
        f"{item['condition']} {item['probability_pct']}"
        for item in record.get("differential_diagnosis", [])[:3]
    )
    initial = record.get("decoded_initial_evidence") or decode_evidence(record.get("initial_evidence"))
    if show_codes:
        initial_text = f"Initial evidence `{initial.get('code', '?')}`={initial.get('meaning', '?')}"
    else:
        initial_text = f"Initial symptom: {initial.get('meaning', '?')}"
    return (
        f"Age {record.get('age', '?')} | Sex {record.get('sex', '?')} | "
        f"{initial_text} | "
        f"Evidences: {evidence_preview}"
        + (f" | Dataset DDx: {diff_preview}" if diff_preview else "")
    )
