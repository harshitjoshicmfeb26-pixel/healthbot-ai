"""
utils/multilingual_normalizer.py
────────────────────────────────
Normalize English, Hinglish, Marathi, Romanized Marathi, and mixed symptom
input into canonical English medical symptom terms.
"""

import re
from typing import Dict, List, Tuple

from utils.language_detector import DEVANAGARI_RE, detect_language
from utils.fuzzy_symptom_matcher import fuzzy_match_symptoms
from utils.medical_synonyms import FILLER_WORDS, all_symptom_maps, merged_symptom_map
from utils.ollama_nlu_extractor import extract_clinical_details


TOKEN_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")


def _latin_phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![a-zA-Z])" + re.escape(phrase) + r"(?![a-zA-Z])", re.IGNORECASE)


def _find_phrase_matches(text: str) -> List[Tuple[str, str]]:
    """Return longest phrase matches as (source_phrase, canonical_symptom)."""
    phrase_map = merged_symptom_map()
    matches: List[Tuple[int, int, str, str]] = []
    occupied: list[tuple[int, int]] = []

    for phrase, canonical in sorted(phrase_map.items(), key=lambda item: len(item[0]), reverse=True):
        if DEVANAGARI_RE.search(phrase):
            search_start = 0
            while True:
                idx = text.find(phrase, search_start)
                if idx == -1:
                    break
                span = (idx, idx + len(phrase))
                if not any(max(span[0], old[0]) < min(span[1], old[1]) for old in occupied):
                    matches.append((span[0], span[1], phrase, canonical))
                    occupied.append(span)
                search_start = idx + len(phrase)
        else:
            for match in _latin_phrase_pattern(phrase).finditer(text):
                span = match.span()
                if not any(max(span[0], old[0]) < min(span[1], old[1]) for old in occupied):
                    matches.append((span[0], span[1], match.group(0), canonical))
                    occupied.append(span)

    return [(source, canonical) for _, _, source, canonical in sorted(matches, key=lambda item: item[0])]


def _unique_preserve_order(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _extract_unmapped_terms(text: str, mapped_sources: List[str]) -> List[str]:
    text_for_terms = text
    for source in mapped_sources:
        if DEVANAGARI_RE.search(source):
            text_for_terms = text_for_terms.replace(source, " ")
        else:
            text_for_terms = _latin_phrase_pattern(source).sub(" ", text_for_terms)

    terms = []
    for token in TOKEN_RE.findall(text_for_terms):
        normalized = token.lower()
        if normalized not in FILLER_WORDS and len(normalized) > 1:
            terms.append(token)
    return _unique_preserve_order(terms)


def _should_try_ollama_nlu(original_text: str, detected: Dict, canonical_terms: List[str], unmapped_terms: List[str]) -> bool:
    if not str(original_text or "").strip():
        return False
    language = (detected or {}).get("language", "unknown")
    if not canonical_terms:
        return True
    return language in {"hinglish", "marathi_devanagari", "romanized_marathi", "mixed"} and bool(unmapped_terms)


def normalize_symptoms(text: str) -> Dict:
    """
    Normalize symptom text into canonical English symptom terms.

    Returns:
        {
          "original_text": "...",
          "detected_language": {...},
          "normalized_text": "...",
          "mapped_symptoms": [{"source": "...", "canonical": "..."}],
          "unmapped_terms": [...]
        }
    """
    original_text = text or ""
    detected = detect_language(original_text)
    matches = _find_phrase_matches(original_text)

    mapped_symptoms = [
        {
            "source": source,
            "canonical": canonical,
            "match_type": "exact",
            "score": 1.0,
        }
        for source, canonical in matches
    ]
    canonical_terms = _unique_preserve_order([canonical for _, canonical in matches])
    mapped_sources = [source for source, _ in matches]

    fuzzy_matches = fuzzy_match_symptoms(
        original_text,
        exact_sources=mapped_sources,
        existing_canonicals=canonical_terms,
    )
    mapped_symptoms.extend(fuzzy_matches)
    canonical_terms = _unique_preserve_order(
        [item["canonical"] for item in mapped_symptoms if item.get("canonical")]
    )
    mapped_sources = [item["source"] for item in mapped_symptoms if item.get("source")]
    unmapped_terms = _extract_unmapped_terms(original_text, mapped_sources)
    llm_extraction = {}

    if _should_try_ollama_nlu(original_text, detected, canonical_terms, unmapped_terms):
        llm_extraction = extract_clinical_details(original_text)
        if llm_extraction.get("used"):
            for symptom in llm_extraction.get("symptoms", []):
                if symptom and symptom not in canonical_terms:
                    mapped_symptoms.append({
                        "source": symptom,
                        "canonical": symptom,
                        "match_type": "ollama_nlu",
                        "score": 1.0,
                    })
                    canonical_terms.append(symptom)

    if canonical_terms:
        normalized_text = " ".join(canonical_terms)
    else:
        normalized_text = original_text

    return {
        "original_text": original_text,
        "detected_language": detected,
        "normalized_text": normalized_text,
        "mapped_symptoms": mapped_symptoms,
        "unmapped_terms": unmapped_terms,
        "llm_extraction": llm_extraction,
    }


def normalize_to_english_text(text: str) -> str:
    """Return only normalized English text for model preprocessing."""
    return normalize_symptoms(text).get("normalized_text", text or "")
