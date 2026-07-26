"""
utils/ollama_nlu_extractor.py
──────────────────────────────
Optional local Ollama JSON extractor for multilingual clinical text.

This module is deliberately NLU-only. It extracts symptoms and chat slots
from Hindi, Hinglish, Marathi, Romanized Marathi, or mixed text, then validates
the extracted symptoms against the project's known canonical symptom list.
It never predicts disease and it never changes model probabilities directly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

try:
    import requests
except Exception:  # pragma: no cover - requests is pinned in requirements
    requests = None

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_NLU_ENABLED,
    OLLAMA_NLU_MODEL,
    OLLAMA_NLU_TIMEOUT,
)
from utils.medical_synonyms import canonical_symptoms, merged_symptom_map


SYSTEM_PROMPT = """You extract structured clinical details from user text.
Return JSON only. Do not diagnose. Do not name diseases.
Normalize symptoms to canonical English symptom phrases when possible.
Use null when a field is not present.

Schema:
{
  "symptoms": ["fever", "cough"],
  "age": 35,
  "gender": "M",
  "duration": "2 days",
  "severity": "moderate",
  "pain_location": "head",
  "previous_disease": "diabetes",
  "family_history": "none",
  "language": "hinglish"
}
"""

_GENDER_MAP = {
    "m": "M",
    "male": "M",
    "man": "M",
    "boy": "M",
    "f": "F",
    "female": "F",
    "woman": "F",
    "girl": "F",
}
_SEVERITY_VALUES = {"mild", "moderate", "severe"}
_DURATION_RE = re.compile(
    r"\b\d+\s*(?:minute|minutes|min|hour|hours|hr|hrs|day|days|week|weeks|month|months|year|years)\b"
    r"|\b(?:since|from)\s+(?:yesterday|today|this morning|morning|last night|last week|last month)\b",
    re.IGNORECASE,
)


def ollama_nlu_status() -> Dict[str, Any]:
    return {
        "enabled": bool(OLLAMA_NLU_ENABLED),
        "model": OLLAMA_NLU_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "timeout": OLLAMA_NLU_TIMEOUT,
        "available": bool(OLLAMA_NLU_ENABLED and requests is not None),
    }


def _empty_result(reason: str) -> Dict[str, Any]:
    return {
        "enabled": bool(OLLAMA_NLU_ENABLED),
        "used": False,
        "model": OLLAMA_NLU_MODEL,
        "symptoms": [],
        "slots": {},
        "reason": reason,
    }


def _json_from_text(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start:end + 1])


def _clean_short_text(value: Any, max_len: int = 160) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text or text.lower() in {"null", "none", "unknown", "not mentioned"}:
        return None
    return text[:max_len]


def _clean_age(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\b(\d{1,3})\b", str(value))
    if not match:
        return None
    age = int(match.group(1))
    return age if 0 <= age <= 120 else None


def _clean_gender(value: Any) -> str | None:
    if value is None:
        return None
    return _GENDER_MAP.get(str(value).strip().lower())


def _clean_duration(value: Any) -> str | None:
    text = _clean_short_text(value, max_len=60)
    if not text:
        return None
    match = _DURATION_RE.search(text)
    return match.group(0) if match else text


def _clean_severity(value: Any) -> str | None:
    text = _clean_short_text(value, max_len=40)
    if not text:
        return None
    lowered = text.lower()
    if lowered in _SEVERITY_VALUES:
        return lowered
    score_match = re.search(r"\b([1-9]|10)\s*(?:/10|out of 10)?\b", lowered)
    if score_match:
        return f"{score_match.group(1)}/10"
    return None


def _clean_symptoms(values: Any) -> List[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []

    phrase_map = merged_symptom_map()
    allowed = canonical_symptoms()
    symptoms: List[str] = []
    for value in raw_values:
        text = _clean_short_text(value, max_len=80)
        if not text:
            continue
        lowered = text.lower()
        canonical = phrase_map.get(lowered, lowered)
        if canonical in allowed and canonical not in symptoms:
            symptoms.append(canonical)
    return symptoms


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    slots = {
        "age": _clean_age(payload.get("age")),
        "gender": _clean_gender(payload.get("gender")),
        "duration": _clean_duration(payload.get("duration")),
        "severity": _clean_severity(payload.get("severity")),
        "pain_location": _clean_short_text(payload.get("pain_location"), max_len=80),
        "previous_disease": _clean_short_text(payload.get("previous_disease"), max_len=160),
        "family_history": _clean_short_text(payload.get("family_history"), max_len=160),
    }
    slots = {key: value for key, value in slots.items() if value is not None}
    return {
        "enabled": bool(OLLAMA_NLU_ENABLED),
        "used": True,
        "model": OLLAMA_NLU_MODEL,
        "symptoms": _clean_symptoms(payload.get("symptoms")),
        "slots": slots,
        "language": _clean_short_text(payload.get("language"), max_len=40),
    }


def extract_clinical_details(text: str) -> Dict[str, Any]:
    """
    Extract symptoms and slots with local Ollama, then validate the result.

    Returns a structured dict with `used=False` when disabled/unavailable or
    when the local model response cannot be parsed safely.
    """
    if not OLLAMA_NLU_ENABLED:
        return _empty_result("disabled")
    if requests is None:
        return _empty_result("requests_unavailable")
    if not str(text or "").strip():
        return _empty_result("empty_text")

    allowed = ", ".join(sorted(canonical_symptoms()))
    prompt = (
        SYSTEM_PROMPT
        + "\nAllowed canonical symptoms:\n"
        + allowed
        + "\n\nUser text:\n"
        + str(text)
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_NLU_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "top_p": 0.2},
            },
            timeout=OLLAMA_NLU_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        payload = _json_from_text(data.get("response", ""))
    except Exception as exc:
        result = _empty_result("ollama_error")
        result["error"] = str(exc)
        return result

    result = _sanitize_payload(payload)
    if not result["symptoms"] and not result["slots"]:
        result["used"] = False
        result["reason"] = "no_valid_extraction"
    return result
