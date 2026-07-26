"""
utils/language_detector.py
──────────────────────────
Lightweight language/script detection for healthcare symptom text.

Supported labels:
  - english
  - hinglish
  - marathi_devanagari
  - romanized_marathi
  - mixed
"""

import re
from typing import Dict, List

from utils.medical_synonyms import (
    ENGLISH_SYMPTOM_MAP,
    HINGLISH_SYMPTOM_MAP,
    MARATHI_SYMPTOM_MAP,
    ROMAN_MARATHI_SYMPTOM_MAP,
)


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _contains_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def _count_devanagari(text: str) -> int:
    return len(DEVANAGARI_RE.findall(text or ""))


def _match_keywords(text: str, phrases: List[str], ignore_case: bool = True) -> List[str]:
    flags = re.IGNORECASE if ignore_case else 0
    matches: List[str] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        if DEVANAGARI_RE.search(phrase):
            if phrase in text:
                matches.append(phrase)
        else:
            pattern = r"(?<![a-zA-Z])" + re.escape(phrase) + r"(?![a-zA-Z])"
            if re.search(pattern, text, flags=flags):
                matches.append(phrase)
    return matches


def detect_language(text: str) -> Dict:
    """
    Detect likely language/script from symptom text.

    Returns:
        {
          "language": "...",
          "script": "...",
          "confidence": 0.0-1.0,
          "matched_keywords": [...]
        }
    """
    text = text or ""
    stripped = text.strip()
    if not stripped:
        return {
            "language": "unknown",
            "script": "unknown",
            "confidence": 0.0,
            "matched_keywords": [],
        }

    dev_count = _count_devanagari(stripped)
    has_dev = dev_count > 0

    english_matches = _match_keywords(stripped, list(ENGLISH_SYMPTOM_MAP.keys()))
    hinglish_matches = _match_keywords(stripped, list(HINGLISH_SYMPTOM_MAP.keys()))
    marathi_matches = _match_keywords(stripped, list(MARATHI_SYMPTOM_MAP.keys()), ignore_case=False)
    roman_marathi_matches = _match_keywords(stripped, list(ROMAN_MARATHI_SYMPTOM_MAP.keys()))

    latin_signal = bool(english_matches or hinglish_matches or roman_marathi_matches)
    scores = {
        "english": len(english_matches),
        "hinglish": len(hinglish_matches),
        "marathi_devanagari": len(marathi_matches) + (2 if has_dev else 0),
        "romanized_marathi": len(roman_marathi_matches),
    }

    matched_keywords = list(dict.fromkeys(
        english_matches + hinglish_matches + marathi_matches + roman_marathi_matches
    ))

    if has_dev and latin_signal:
        language = "mixed"
        script = "mixed"
        confidence = min(0.95, 0.65 + 0.05 * len(matched_keywords))
    elif has_dev:
        language = "marathi_devanagari"
        script = "devanagari"
        confidence = min(0.95, 0.70 + 0.05 * len(marathi_matches))
    else:
        best_language = max(scores, key=scores.get)
        best_score = scores[best_language]

        if best_score == 0:
            language = "english"
            script = "latin"
            confidence = 0.45
        elif best_language in {"hinglish", "romanized_marathi"} and scores["english"] > 0:
            language = "mixed"
            script = "latin"
            confidence = min(0.90, 0.60 + 0.05 * best_score)
        else:
            language = best_language
            script = "latin"
            confidence = min(0.92, 0.62 + 0.06 * best_score)

    return {
        "language": language,
        "script": script,
        "confidence": round(confidence, 2),
        "matched_keywords": matched_keywords,
    }


def has_devanagari(text: str) -> bool:
    """Public helper for other modules."""
    return _contains_devanagari(text)
