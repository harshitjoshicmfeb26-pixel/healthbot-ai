"""
utils/fuzzy_symptom_matcher.py
──────────────────────────────
Generic fuzzy symptom phrase matcher.

Exact multilingual phrase matching should run first. This module then tries to
map remaining text spans to known symptom phrases using string similarity and
scikit-fuzzy membership grading when available.

This is still symptom normalization, not disease prediction.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np

try:
    import skfuzzy as fuzz

    SKFUZZY_AVAILABLE = True
except Exception:
    fuzz = None
    SKFUZZY_AVAILABLE = False

from utils.language_detector import DEVANAGARI_RE
from utils.medical_synonyms import FILLER_WORDS, all_symptom_maps


TOKEN_RE = re.compile(r"[\u0900-\u097Fa-zA-Z]+")


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z\u0900-\u097f\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase_pattern(phrase: str) -> re.Pattern:
    return re.compile(r"(?<![a-zA-Z])" + re.escape(phrase) + r"(?![a-zA-Z])", re.IGNORECASE)


def _remove_exact_sources(text: str, exact_sources: List[str]) -> str:
    remaining = text
    for source in exact_sources:
        if DEVANAGARI_RE.search(source):
            remaining = remaining.replace(source, " ")
        else:
            remaining = _phrase_pattern(source).sub(" ", remaining)
    return remaining


def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    tokens = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.lower() in FILLER_WORDS:
            continue
        tokens.append((token, match.start(), match.end()))
    return tokens


def _candidate_phrases() -> List[Dict]:
    candidates = []
    for language, symptom_map in all_symptom_maps().items():
        for phrase, canonical in symptom_map.items():
            phrase_norm = _normalize_for_match(phrase)
            if phrase_norm:
                candidates.append({
                    "language": language,
                    "phrase": phrase,
                    "phrase_norm": phrase_norm,
                    "canonical": canonical,
                    "token_count": len(phrase_norm.split()),
                })
    return candidates


def _similarity_score(source: str, candidate: str) -> float:
    source_norm = _normalize_for_match(source)
    candidate_norm = _normalize_for_match(candidate)
    if not source_norm or not candidate_norm:
        return 0.0

    source_tokens = set(source_norm.split())
    candidate_tokens = set(candidate_norm.split())
    source_token_list = source_norm.split()
    candidate_token_list = candidate_norm.split()
    overlap = len(source_tokens & candidate_tokens)
    token_score = (
        2 * overlap / (len(source_tokens) + len(candidate_tokens))
        if source_tokens and candidate_tokens
        else 0.0
    )
    char_score = SequenceMatcher(None, source_norm, candidate_norm).ratio()
    token_count_balance = (
        min(len(source_token_list), len(candidate_token_list))
        / max(len(source_token_list), len(candidate_token_list))
        if source_token_list and candidate_token_list
        else 0.0
    )

    def _best_token_average(left: list[str], right: list[str]) -> float:
        if not left or not right:
            return 0.0
        scores = [
            max(SequenceMatcher(None, token, other).ratio() for other in right)
            for token in left
        ]
        return float(sum(scores) / len(scores))

    fuzzy_token_score = (
        (_best_token_average(source_token_list, candidate_token_list)
         + _best_token_average(candidate_token_list, source_token_list))
        / 2
    ) * token_count_balance

    if source_norm == candidate_norm:
        return 1.0

    containment_score = 0.0
    if (
        len(source_norm) >= 4
        and len(candidate_norm) >= 4
        and len(source_tokens) >= 2
        and len(candidate_tokens) >= 2
    ):
        if source_norm in candidate_norm or candidate_norm in source_norm:
            containment_score = 0.90

    blended_score = (0.50 * char_score) + (0.25 * token_score) + (0.25 * fuzzy_token_score)
    return max(blended_score, containment_score)


def fuzzy_confidence(score: float) -> Dict:
    """
    Grade a 0-1 similarity score using scikit-fuzzy membership functions.

    If scikit-fuzzy is not installed, return deterministic threshold labels.
    """
    score = max(0.0, min(1.0, float(score)))
    if SKFUZZY_AVAILABLE:
        universe = np.arange(0, 101, 1)
        score_pct = score * 100
        low = fuzz.trapmf(universe, [0, 0, 55, 70])
        medium = fuzz.trimf(universe, [60, 75, 90])
        high = fuzz.trapmf(universe, [80, 88, 100, 100])

        low_degree = float(fuzz.interp_membership(universe, low, score_pct))
        medium_degree = float(fuzz.interp_membership(universe, medium, score_pct))
        high_degree = float(fuzz.interp_membership(universe, high, score_pct))
        memberships = {
            "low": low_degree,
            "medium": medium_degree,
            "high": high_degree,
        }
        label = max(memberships, key=memberships.get)
        return {
            "label": label,
            "score": round(score, 3),
            "memberships": {key: round(value, 3) for key, value in memberships.items()},
            "method": "scikit-fuzzy",
        }

    if score >= 0.86:
        label = "high"
    elif score >= 0.74:
        label = "medium"
    else:
        label = "low"
    return {
        "label": label,
        "score": round(score, 3),
        "memberships": {},
        "method": "threshold-fallback",
    }


def fuzzy_match_symptoms(
    text: str,
    exact_sources: List[str] | None = None,
    existing_canonicals: List[str] | None = None,
    threshold: float = 0.78,
    max_ngram: int = 5,
) -> List[Dict]:
    """
    Match remaining text spans to canonical symptom terms.

    Returns dictionaries with:
        source, canonical, matched_phrase, score, confidence, match_type, start, end
    """
    exact_sources = exact_sources or []
    existing_canonicals = existing_canonicals or []
    remaining_text = _remove_exact_sources(text or "", exact_sources)
    tokens = _tokenize_with_spans(remaining_text)
    if not tokens:
        return []

    candidates = _candidate_phrases()
    possible_matches: List[Dict] = []
    max_window = min(max_ngram, max((c["token_count"] for c in candidates), default=1))

    for start_idx in range(len(tokens)):
        for end_idx in range(start_idx + 1, min(len(tokens), start_idx + max_window) + 1):
            span_tokens = tokens[start_idx:end_idx]
            source_text = " ".join(token for token, _, _ in span_tokens)
            source_token_count = len(source_text.split())
            span_start = span_tokens[0][1]
            span_end = span_tokens[-1][2]

            for candidate in candidates:
                if abs(source_token_count - candidate["token_count"]) > 2:
                    continue
                score = _similarity_score(source_text, candidate["phrase_norm"])
                if score < threshold:
                    continue
                confidence = fuzzy_confidence(score)
                if confidence["label"] == "low":
                    continue
                possible_matches.append({
                    "source": source_text,
                    "canonical": candidate["canonical"],
                    "matched_phrase": candidate["phrase"],
                    "matched_language": candidate["language"],
                    "score": confidence["score"],
                    "confidence": confidence,
                    "match_type": "fuzzy",
                    "start": span_start,
                    "end": span_end,
                })

    selected: List[Dict] = []
    occupied: List[Tuple[int, int]] = []
    seen_canonicals = set(existing_canonicals)
    for match in sorted(possible_matches, key=lambda item: (item["score"], item["end"] - item["start"]), reverse=True):
        span = (match["start"], match["end"])
        overlaps = any(max(span[0], old[0]) < min(span[1], old[1]) for old in occupied)
        if overlaps:
            continue
        if match["canonical"] in seen_canonicals:
            continue
        selected.append(match)
        occupied.append(span)
        seen_canonicals.add(match["canonical"])

    return sorted(selected, key=lambda item: item["start"])
