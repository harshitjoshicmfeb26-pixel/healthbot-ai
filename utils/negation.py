"""
utils/negation.py
──────────────────
Lightweight, dependency-free negation and uncertainty detection.

This is a deliberately small implementation of the same idea behind the
NegEx / ConText algorithms used in clinical NLP (Chapman et al., 2001):

    1. Find negation trigger phrases ("no", "denies", "ruled out", ...).
    2. Mark a bounded scope around each trigger as "negated".
    3. Stop the scope early at punctuation, conjunctions ("but", "however"),
       or a second trigger — so "no fever but has chest pain" does not mark
       "chest pain" as negated.
    4. Skip pseudo-negations ("not ruled out", "cannot rule out") which look
       like negation triggers but actually mean the opposite.

Why not pull in scispaCy / medspaCy / negspaCy for this:
    Those are excellent, more rigorous tools, but they pull in spaCy plus a
    biomedical model (several hundred MB) for a project that otherwise runs
    on plain regex/dict lookups and is meant to stay light enough to run on
    a CPU-only, memory-constrained machine. This module covers the same core
    mechanism (trigger + scoped window + pseudo-negation guard) for the
    English / Hinglish / Marathi phrasing this project already supports,
    with zero new dependencies. If the project later adopts a heavier
    clinical-NLP stack, this module's public functions
    (`negated_ranges`, `is_negated`, `filter_negated`) are the seam to swap.

This module never decides clinical meaning on its own — callers are
responsible for combining negation status with their own evidence/red-flag
logic. It only answers: "is the text at this position inside a negated
scope?"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# ── Trigger phrases ──────────────────────────────────────────────────────
# Longest-first matching is applied at use-time so "no evidence of" wins
# over the shorter "no".

PRE_NEGATION_TRIGGERS: tuple[str, ...] = (
    # English only — Hindi/Marathi negation particles grammatically follow
    # the word they negate ("bukhar nahi hai" = "no fever"), so they live in
    # POST_NEGATION_TRIGGERS below, not here.
    "no evidence of", "no sign of", "no signs of", "no history of",
    "no longer has", "absence of", "free of", "negative for",
    "denies having", "denies", "denied", "without any", "without",
    "not experiencing", "not having", "not feeling", "not", "no",
    # Explicit experiencer/possession contractions.  These are deliberately
    # phrase-scoped rather than a blanket ``n't`` rule: "can't breathe" and
    # "couldn't walk" describe an inability, not absence of the symptom.
    "doesn't have", "didn't have", "don't have",
    "hasn't had", "hadn't had", "haven't had",
    "wasn't experiencing", "weren't experiencing",
    "isn't experiencing", "aren't experiencing",
    "isn't", "aren't", "wasn't", "weren't",
    "doesn't feel", "didn't feel", "don't feel",
    "hasn't experienced", "hadn't experienced", "haven't experienced",
    "wasn't feeling", "weren't feeling", "isn't feeling", "aren't feeling",
    # The DDXPlus text normalizer strips apostrophes, so retain equivalent
    # tokenized forms ("don't" -> "don t") for callers using normalized text.
    "doesn t have", "didn t have", "don t have",
    "hasn t had", "hadn t had", "haven t had",
    "wasn t experiencing", "weren t experiencing",
    "isn t experiencing", "aren t experiencing",
    "isn t", "aren t", "wasn t", "weren t",
    "doesn t feel", "didn t feel", "don t feel",
    "hasn t experienced", "hadn t experienced", "haven t experienced",
    "wasn t feeling", "weren t feeling", "isn t feeling", "aren t feeling",
)

POST_NEGATION_TRIGGERS: tuple[str, ...] = (
    # English
    "ruled out", "was negative", "has resolved", "is resolved",
    "resolved", "negative", "denied",
    # Hinglish / romanized Marathi — negation particle follows the symptom
    "koi nahi", "bilkul nahi", "nahi hai", "nahi tha", "nahin", "nahi",
    # Marathi (Devanagari) — same trailing-negation grammar
    "नाहीये", "नाही",
)

# Phrases that contain a negation trigger as a substring but do not actually
# negate what follows/precedes them (classic NegEx "pseudo-negation" list).
PSEUDO_NEGATION_TRIGGERS: tuple[str, ...] = (
    "not ruled out", "cannot rule out", "can not rule out",
    "not certain if", "not certain whether", "not sure if",
    "not only", "no increase", "no further increase", "no change",
)

# Phrases that end a negation scope early (the negation does not "jump
# across" these into the next clause).
TERMINATION_TOKENS: tuple[str, ...] = (
    "but", "however", "except", "apart from", "aside from",
    "though", "although", "yet", "still", ";", ".", ",", "and also",
    # Hindi/Hinglish and Marathi "but"
    "par", "lekin", "पण",
)

# These markers are intentionally narrow.  They support an explicit later
# assertion or resolution in the immediately following clause, rather than
# attempting unrestricted temporal/coreference reasoning.
_TEMPORAL_REASSERTION_RE = re.compile(
    r"\b(?:but|however)\b.*?"
    r"(?:\b(?:started|began|returned|came back|is back)\b|"
    r"\b(?:now|today|currently|later|this morning|this evening)\b.*?"
    r"\b(?:i have|i do|it started|it began|it returned|it came back)\b|"
    r"\b(?:i have|i do)\b.*?\b(?:now|today|currently|later|this morning|this evening)\b)",
    re.IGNORECASE,
)
_TEMPORAL_RETURN_RE = re.compile(
    r"\b(?:returned|came back|is back)\b.*?"
    r"\b(?:now|today|currently|later|this morning|this evening)\b",
    re.IGNORECASE,
)
_TEMPORAL_RESOLUTION_RE = re.compile(
    r"\b(?:but|however)\b.*?"
    r"(?:\b(?:don t have|doesn t have|didn t have|haven t|hasn t|"
    r"isn t|aren t|wasn t|weren t)\b.*?\b(?:now|currently|today)\b|"
    r"\b(?:gone|resolved|ended|stopped|no longer)\b)",
    re.IGNORECASE,
)
_RESOLVED_RE = re.compile(
    r"\b(?:has|have|is|was)\s+resolved\b|\b(?:gone|resolved|ended|stopped)\b",
    re.IGNORECASE,
)

# Maximum number of words a pre-trigger's scope extends over before it is
# considered to have run out (mirrors NegEx's default ~5-7 token window).
DEFAULT_SCOPE_WORDS = 6

_WORD_RE = re.compile(r"\S+")


def _normalize_apostrophes(text: str) -> str:
    """Normalize common Unicode apostrophes without changing text length."""
    return text.replace("\u2018", "'").replace("\u2019", "'").replace("\u02bc", "'")


def _temporal_text(text: str) -> str:
    """Normalize only punctuation needed by the narrow temporal rules."""
    normalized = _normalize_apostrophes(str(text or "")).lower().replace("'", " ")
    return re.sub(r"[^a-z0-9\u0900-\u097f ]+", " ", normalized)


def temporal_reassertion(text: str, phrase: str) -> bool:
    """Return whether *phrase* is explicitly reasserted in a later clause."""
    raw = _temporal_text(text)
    target = _temporal_text(phrase).strip()
    if not raw or not target:
        return False
    first = re.search(r"(?<!\S)" + re.escape(target) + r"(?!\S)", raw)
    if first is None:
        return False
    suffix = raw[first.end():]
    return bool(_TEMPORAL_REASSERTION_RE.search(suffix) or _TEMPORAL_RETURN_RE.search(suffix))


def temporal_resolution(text: str, phrase: str) -> bool:
    """Return whether *phrase* is explicitly resolved in a later clause."""
    raw = _temporal_text(text)
    target = _temporal_text(phrase).strip()
    if not raw or not target:
        return False
    first = re.search(r"(?<!\S)" + re.escape(target) + r"(?!\S)", raw)
    if first is None:
        return False
    prefix = raw[max(0, first.start() - 32):first.start()]
    if re.search(r"no longer\s+(?:has|have)\s*$", prefix, re.IGNORECASE):
        return True
    suffix = raw[first.end():]
    if _TEMPORAL_RESOLUTION_RE.search(suffix):
        return True
    clause = re.split(r"\b(?:but|however)\b", suffix, maxsplit=1, flags=re.IGNORECASE)
    return len(clause) == 2 and bool(_RESOLVED_RE.search(clause[1]))


@dataclass(frozen=True)
class NegationSpan:
    start: int
    end: int
    trigger: str
    direction: str  # "pre" or "post"


def _compile_phrase_pattern(phrase: str) -> re.Pattern:
    if re.search(r"[\u0900-\u097F]", phrase):
        return re.compile(re.escape(phrase))
    return re.compile(r"(?<![a-zA-Z])" + re.escape(phrase) + r"(?![a-zA-Z])", re.IGNORECASE)


def _find_all(text: str, phrases: Sequence[str]) -> list[tuple[int, int, str]]:
    """Find all non-overlapping matches of any phrase, longest-first."""
    found: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        for match in _compile_phrase_pattern(phrase).finditer(text):
            span = match.span()
            if any(max(span[0], a) < min(span[1], b) for a, b in occupied):
                continue
            found.append((span[0], span[1], phrase))
            occupied.append(span)
    return sorted(found, key=lambda item: item[0])


def _is_pseudo_negation(text: str, trigger_start: int, trigger_end: int) -> bool:
    window = text[max(0, trigger_start - 20): trigger_end + 20]
    return any(pseudo in window.lower() for pseudo in PSEUDO_NEGATION_TRIGGERS)


def _next_word_boundaries(text: str, start: int, max_words: int) -> int:
    """Return the end offset after `max_words` words starting at `start`."""
    count = 0
    end = start
    for match in _WORD_RE.finditer(text, start):
        if count >= max_words:
            break
        end = match.end()
        count += 1
    return end if count else len(text)


def _scope_end(text: str, trigger_end: int, max_words: int) -> int:
    """Pre-trigger scope: extends forward, cut short by terminators."""
    candidate_end = _next_word_boundaries(text, trigger_end, max_words)
    earliest_terminator = candidate_end
    for terminator in TERMINATION_TOKENS:
        idx = text.find(terminator, trigger_end, candidate_end)
        if idx != -1:
            earliest_terminator = min(earliest_terminator, idx)
    return earliest_terminator


def _scope_start(text: str, trigger_start: int, max_words: int) -> int:
    """Post-trigger scope: extends backward, cut short by terminators."""
    words_before = list(_WORD_RE.finditer(text[:trigger_start]))
    if len(words_before) <= max_words:
        candidate_start = 0
    else:
        candidate_start = words_before[-max_words].start()
    latest_terminator = candidate_start
    for terminator in TERMINATION_TOKENS:
        idx = text.rfind(terminator, candidate_start, trigger_start)
        if idx != -1:
            latest_terminator = max(latest_terminator, idx + len(terminator))
    return latest_terminator


def negated_ranges(text: str, scope_words: int = DEFAULT_SCOPE_WORDS) -> list[NegationSpan]:
    """
    Return the negated character ranges in `text`.

    Both pre-triggers ("no fever") and post-triggers ("fever was ruled out")
    are handled. Pseudo-negations ("not ruled out") are skipped.
    """
    raw_text = _normalize_apostrophes(str(text or ""))
    if not raw_text.strip():
        return []

    spans: list[NegationSpan] = []

    for start, end, trigger in _find_all(raw_text, PRE_NEGATION_TRIGGERS):
        if _is_pseudo_negation(raw_text, start, end):
            continue
        scope_end = _scope_end(raw_text, end, scope_words)
        if scope_end > end:
            spans.append(NegationSpan(end, scope_end, trigger, "pre"))

    for start, end, trigger in _find_all(raw_text, POST_NEGATION_TRIGGERS):
        if _is_pseudo_negation(raw_text, start, end):
            continue
        scope_start = _scope_start(raw_text, start, scope_words)
        if scope_start < start:
            spans.append(NegationSpan(scope_start, start, trigger, "post"))

    return spans


def is_negated(text: str, phrase_start: int, phrase_end: int, scope_words: int = DEFAULT_SCOPE_WORDS) -> bool:
    """Is the span [phrase_start, phrase_end) inside a negated scope?"""
    phrase = str(text or "")[phrase_start:phrase_end]
    if temporal_resolution(text, phrase):
        return True
    if temporal_reassertion(text, phrase):
        return False
    for span in negated_ranges(text, scope_words):
        if max(phrase_start, span.start) < min(phrase_end, span.end):
            return True
    return False


def find_phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    """Find the first character span of `phrase` inside `text`, or None."""
    match = _compile_phrase_pattern(phrase).search(text)
    return match.span() if match else None


def filter_negated(text: str, phrases: Iterable[str]) -> tuple[list[str], list[str]]:
    """
    Split `phrases` (each expected to literally appear in `text`) into
    (affirmed, negated) based on whether their first occurrence in `text`
    falls inside a negated scope.
    """
    affirmed: list[str] = []
    denied: list[str] = []
    spans = negated_ranges(text)
    for phrase in phrases:
        located = find_phrase_span(text, phrase)
        if located is None:
            affirmed.append(phrase)
            continue
        start, end = located
        if any(max(start, span.start) < min(end, span.end) for span in spans):
            denied.append(phrase)
        else:
            affirmed.append(phrase)
    return affirmed, denied
