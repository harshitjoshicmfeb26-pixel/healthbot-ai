"""
utils/red_flag_rules.py
───────────────────────
Rule-based clinical red-flag triage.

This module does not diagnose emergency disease. It only detects warning signs
that should be prioritized before ordinary model predictions.
"""

import re
from typing import Dict, List

from utils.multilingual_normalizer import normalize_symptoms
from utils.negation import is_negated


RED_FLAG_PATTERNS = [
    ("severe chest pain", "severe chest pain", "emergency"),
    ("chest pain", "chest pain", "urgent"),
    ("difficulty breathing", "difficulty breathing", "emergency"),
    ("shortness of breath", "shortness of breath", "emergency"),
    ("difficulty urinating", "difficulty urinating", "urgent"),
    ("urinary retention", "difficulty urinating", "urgent"),
    ("cannot urinate", "difficulty urinating", "urgent"),
    ("unconsciousness", "unconsciousness", "emergency"),
    ("unconscious", "unconsciousness", "emergency"),
    ("fainting", "fainting", "urgent"),
    ("sudden weakness", "sudden weakness", "emergency"),
    ("facial drooping", "facial drooping", "emergency"),
    ("slurred speech", "slurred speech", "emergency"),
    ("confusion", "confusion", "urgent"),
    ("severe headache", "severe headache", "urgent"),
    ("blood in vomit", "blood in vomit", "emergency"),
    ("vomiting blood", "blood in vomit", "emergency"),
    ("blood in stool", "blood in stool", "urgent"),
    ("severe dehydration", "severe dehydration", "urgent"),
    ("persistent vomiting", "persistent vomiting", "urgent"),
    ("seizure", "seizure", "emergency"),
    ("pregnancy with bleeding", "pregnancy with bleeding", "emergency"),
    ("severe abdominal pain", "severe abdominal pain", "urgent"),
    ("blue lips", "blue lips", "emergency"),
    ("severe allergic reaction", "severe allergic reaction", "emergency"),
    ("swelling of face with breathing difficulty", "face swelling with breathing difficulty", "emergency"),
]

CHILD_TERMS = {"child", "baby", "infant", "kid", "toddler", "लहान", "बाळ", "मुलगा", "मुलगी"}


def _phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    if re.search(r"[\u0900-\u097F]", phrase):
        idx = text.find(phrase)
        return (idx, idx + len(phrase)) if idx != -1 else None
    match = re.search(r"(?<![a-zA-Z])" + re.escape(phrase) + r"(?![a-zA-Z])", text, re.IGNORECASE)
    return match.span() if match else None


def _safety_message(urgency_level: str, matched_flags: List[str]) -> str:
    if urgency_level == "emergency":
        return (
            "Emergency warning sign detected: "
            + ", ".join(matched_flags)
            + ". Please seek urgent in-person medical care or local emergency help now. "
            "This tool is not a medical diagnosis."
        )
    if urgency_level == "urgent":
        return (
            "Important warning sign detected: "
            + ", ".join(matched_flags)
            + ". Please contact a qualified healthcare professional promptly. "
            "This tool is not a medical diagnosis."
        )
    return (
        "No emergency red flag was detected from the provided text. "
        "Continue monitoring symptoms and consult a healthcare professional if symptoms worsen."
    )


def detect_red_flags(text: str) -> Dict:
    """
    Detect red-flag warning signs.

    Negation-aware: a phrase match inside a negated scope ("no chest pain",
    "chest pain was ruled out", "chest pain nahi hai") is not counted as a
    positive red flag. This does not change `urgency_level` to anything more
    reassuring than the rules already produce — it only stops a denied
    symptom from being counted as present.

    Returns:
        {
          "has_red_flag": true/false,
          "matched_flags": [...],
          "denied_flags": [...],
          "urgency_level": "emergency" / "urgent" / "routine",
          "safety_message": "..."
        }
    """
    original_text = text or ""
    normalized = normalize_symptoms(original_text).get("normalized_text", "")
    combined = f"{original_text} {normalized}".lower()

    matched: List[str] = []
    denied: List[str] = []
    urgency = "routine"

    for phrase, label, level in RED_FLAG_PATTERNS:
        span = _phrase_span(combined, phrase)
        if span is None:
            continue
        if is_negated(combined, span[0], span[1]):
            denied.append(label)
            continue
        matched.append(label)
        if level == "emergency":
            urgency = "emergency"
        elif level == "urgent" and urgency != "emergency":
            urgency = "urgent"

    has_child_term = any(term.lower() in combined for term in CHILD_TERMS)
    has_high_fever = "high fever" in combined or "fever" in combined and ("child" in combined or "baby" in combined)
    if has_child_term and has_high_fever and not is_negated(combined, *(_phrase_span(combined, "fever") or (0, 0))):
        matched.append("high fever in child")
        if urgency != "emergency":
            urgency = "urgent"

    has_back_pain = any(phrase in combined for phrase in ["back pain", "lower back pain", "flank pain"])
    has_urine_difficulty = any(
        phrase in combined
        for phrase in ["difficulty urinating", "urinary retention", "cannot urinate", "unable to urinate"]
    )
    if has_back_pain and has_urine_difficulty:
        matched.append("back pain with difficulty urinating")
        if urgency != "emergency":
            urgency = "urgent"

    matched = list(dict.fromkeys(matched))
    denied = list(dict.fromkeys(denied))
    return {
        "has_red_flag": bool(matched),
        "matched_flags": matched,
        "denied_flags": denied,
        "urgency_level": urgency,
        "safety_message": _safety_message(urgency, matched),
    }
