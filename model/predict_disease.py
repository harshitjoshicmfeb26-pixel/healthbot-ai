"""
Prediction helper for the simplified supervised disease classifier.

This is the non-RAG path: collected chat fields are combined into one text
feature and a supervised classifier predicts the disease directly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from config import SIMPLIFIED_DISEASE_METADATA_PATH, SIMPLIFIED_DISEASE_MODEL_PATH


_model = None
_metadata = None


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "unknown":
        return "unknown"
    score_match = re.search(r"\b([1-9]|10)\s*(?:/10|out of 10)?\b", text)
    if score_match:
        score = int(score_match.group(1))
        return f"pain intensity: {score}"
    if text in {"mild", "low"}:
        return "pain intensity: 2"
    if text in {"moderate", "medium"}:
        return "pain intensity: 5"
    if text in {"severe", "high"}:
        return "pain intensity: 8"
    return text


def disease_classifier_available() -> bool:
    return Path(SIMPLIFIED_DISEASE_MODEL_PATH).exists()


def _ensure_loaded():
    global _model, _metadata
    if _model is not None:
        return
    if not disease_classifier_available():
        raise FileNotFoundError(
            f"Simplified disease classifier not found: {SIMPLIFIED_DISEASE_MODEL_PATH}. "
            "Run `python -m model.train_disease_classifier` first."
        )
    _model = joblib.load(SIMPLIFIED_DISEASE_MODEL_PATH)
    if Path(SIMPLIFIED_DISEASE_METADATA_PATH).exists():
        _metadata = json.loads(Path(SIMPLIFIED_DISEASE_METADATA_PATH).read_text(encoding="utf-8"))
    else:
        _metadata = {}


def disease_classifier_metadata() -> dict:
    if not disease_classifier_available():
        return {"available": False, "model_path": str(SIMPLIFIED_DISEASE_MODEL_PATH)}
    _ensure_loaded()
    metadata = dict(_metadata or {})
    metadata["available"] = True
    metadata["model_path"] = str(SIMPLIFIED_DISEASE_MODEL_PATH)
    return metadata


def build_model_input(
    age: Any = "unknown",
    gender: Any = "unknown",
    symptoms_text: Any = "unknown",
    duration: Any = "unknown",
    severity: Any = "unknown",
    pain_location: Any = "unknown",
    previous_disease_or_history: Any = "unknown",
    genetic_or_family_history: Any = "unknown",
) -> str:
    return (
        f"age: {age or 'unknown'} "
        f"gender: {gender or 'unknown'} "
        f"symptoms: {symptoms_text or 'unknown'} "
        f"duration: {duration or 'unknown'} "
        f"severity: {_normalize_severity(severity)} "
        f"pain location: {pain_location or 'unknown'} "
        f"previous disease: {previous_disease_or_history or 'unknown'} "
        f"family history: {genetic_or_family_history or 'unknown'}"
    )


def predict_disease(
    age: Any,
    gender: Any,
    symptoms_text: Any,
    duration: Any = "unknown",
    severity: Any = "unknown",
    pain_location: Any = "unknown",
    previous_disease_or_history: Any = "unknown",
    genetic_or_family_history: Any = "unknown",
    top_n: int = 5,
) -> List[Dict]:
    _ensure_loaded()
    model_input = build_model_input(
        age=age,
        gender=gender,
        symptoms_text=symptoms_text,
        duration=duration,
        severity=severity,
        pain_location=pain_location,
        previous_disease_or_history=previous_disease_or_history,
        genetic_or_family_history=genetic_or_family_history,
    )

    if hasattr(_model, "predict_proba"):
        probabilities = _model.predict_proba([model_input])[0]
    elif hasattr(_model, "decision_function"):
        scores = np.asarray(_model.decision_function([model_input])[0], dtype=float)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / exp_scores.sum()
    else:
        raise AttributeError("Simplified disease classifier has no probability or decision score method.")

    classes = np.asarray(_model.classes_)
    top_indices = np.argsort(probabilities)[::-1][: int(top_n)]
    sorted_scores = np.sort(probabilities)[::-1]
    margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

    rows = []
    for rank, idx in enumerate(top_indices, 1):
        score = float(probabilities[idx])
        rows.append({
            "rank": rank,
            "disease": str(classes[idx]),
            "pathology": str(classes[idx]),
            "label_id": int(idx),
            "confidence": round(score, 4),
            "confidence_pct": f"{score * 100:.1f}%",
            "flag": "HIGH" if score >= 0.60 else "MEDIUM" if score >= 0.30 else "LOW",
            "source": "supervised simplified disease classifier",
            "score_type": "model probability",
            "top_margin": round(margin, 4) if rank == 1 else None,
            "top_margin_pct": f"{margin * 100:.1f}%" if rank == 1 else None,
        })
    return rows
