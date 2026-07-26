"""
DDXPlus metadata decoder.

Loads:
- data/release_evidences.json
- data/release_conditions.json

The loader accepts both this project's generated metadata format and common
DDXPlus-style dict formats. Official metadata can replace the local JSON files
without changing the app code.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from utils.negation import is_negated


BASE_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_PATHS = [
    BASE_DIR / "data" / "release_evidences.json",
    BASE_DIR / "release_evidences.json",
]
CONDITION_PATHS = [
    BASE_DIR / "data" / "release_conditions.json",
    BASE_DIR / "release_conditions.json",
]

EVIDENCE_CODE_RE = re.compile(r"E_\d+(?:_@_(?:V_)?\d+)?", re.IGNORECASE)
CODE_LIKE_NAME_RE = re.compile(r"^E_\d+$", re.IGNORECASE)
QUESTION_ONLY_RE = re.compile(
    r"^\s*(what|why|how|when|where|who|which|can|could|should|do|does|did|is|are|will|would)\b",
    re.IGNORECASE,
)
SEMANTIC_SYMPTOM_HINTS = {
    "ache", "allergy", "back", "belly", "blood", "breath", "breathing", "burning",
    "chest", "chills", "cold", "cough", "dard", "diarrhea", "dizzy", "fever",
    "head", "headache", "itch", "jalan", "khansi", "knee", "nausea", "pain",
    "pee", "peshab", "phlegm", "rash", "saans", "sore", "stomach", "swelling",
    "throat", "urine", "urinating", "vomit", "wheezing", "ताप", "दुख", "खोकला",
    "कफ", "श्वास", "छाती", "लघवी", "जळजळ", "पोट", "डोके", "खाज", "पुरळ",
}

PAIN_CONTEXT_RE = re.compile(
    r"\b(pain|ache|aches|aching|hurt|hurts|hurting|sore|burning|dard|dukht|dukhta|jalan|"
    r"swelling|rash|itching)\b|दुख|दुखत|जळजळ|सूज|खाज|पुरळ",
    re.IGNORECASE,
)
GENERIC_INITIAL_EVIDENCE_CODES = {"E_53"}


CURATED_TEXT_ALIASES = {
    "fever": ["E_91"],
    "high fever": ["E_91"],
    "temperature": ["E_91"],
    "bukhar": ["E_91"],
    "tap": ["E_91"],
    "ताप": ["E_91"],
    "chills": ["E_94"],
    "shivers": ["E_94"],
    "thandi": ["E_94"],
    "thandi lagna": ["E_94"],
    "थंडी": ["E_94"],
    "थंडी वाजते": ["E_94"],
    "sweating": ["E_50"],
    "cough": ["E_201"],
    "coughing": ["E_201"],
    "khansi": ["E_201"],
    "khokla": ["E_201"],
    "खोकला": ["E_201"],
    "intense cough": ["E_203"],
    "coughing fits": ["E_203"],
    "blood in cough": ["E_45"],
    "coughing blood": ["E_45"],
    "cough with blood": ["E_45", "E_201"],
    "sputum": ["E_77"],
    "phlegm": ["E_77"],
    "kaf": ["E_77"],
    "कफ": ["E_77"],
    "कफ आहे": ["E_77"],
    "shortness of breath": ["E_66"],
    "difficulty breathing": ["E_66"],
    "breathlessness": ["E_66"],
    "breathing problem": ["E_66"],
    "saans lene me dikkat": ["E_66"],
    "saans lene mein dikkat": ["E_66"],
    "श्वास घेण्यास त्रास": ["E_66"],
    "out of breath": ["E_64"],
    "wheezing": ["E_214"],
    "wheeze": ["E_214"],
    "ghar ghar": ["E_214"],
    "घरघर": ["E_214"],
    "noisy breathing": ["E_112"],
    "sore throat": ["E_97"],
    "throat pain": ["E_97", "E_53", "E_55_@_V_148"],
    "gala kharab": ["E_97"],
    "gaka kharab": ["E_97"],
    "nausea": ["E_148"],
    "nauseous": ["E_148"],
    "jee michalna": ["E_148"],
    "ji michalna": ["E_148"],
    "jee machalna": ["E_148"],
    "mala mal": ["E_148"],
    "malmal": ["E_148"],
    "मळमळ": ["E_148"],
    "vomiting feeling": ["E_148"],
    "feel like vomiting": ["E_148"],
    "vomiting": ["E_211"],
    "vomited": ["E_211"],
    "diarrhea": ["E_51"],
    "loose motion": ["E_51"],
    "heartburn": ["E_173"],
    "acid reflux": ["E_173"],
    "reflux": ["E_173"],
    "seene mein jalan": ["E_173"],
    "seene me jalan": ["E_173"],
    "chest burning": ["E_173"],
    "burning in stomach": ["E_173"],
    "burning from stomach to throat": ["E_173"],
    "nasal congestion": ["E_181"],
    "seasonal nasal congestion": ["E_181"],
    "blocked nose": ["E_181"],
    "stuffy nose": ["E_181"],
    "runny nose": ["E_181"],
    "dark urine": ["E_188"],
    "urine is dark": ["E_188"],
    "yellow urine": ["E_188"],
    "pale stool": ["E_188"],
    "black stool": ["E_140"],
    "blood in stool": ["E_179"],
    "blood in motion": ["E_179"],
    "body pain": ["E_144"],
    "muscle pain": ["E_144"],
    "diffuse pain": ["E_144"],
    "ang dukhne": ["E_144"],
    "अंग दुखणे": ["E_144"],
    "अंग दुखत": ["E_144"],
    "chest pain": ["E_53", "E_55_@_V_29"],
    "pain in chest": ["E_53", "E_55_@_V_29"],
    "chest me pain": ["E_53", "E_55_@_V_29"],
    "chest mein pain": ["E_53", "E_55_@_V_29"],
    "pain in my chest": ["E_53", "E_55_@_V_29"],
    "my chest hurts": ["E_53", "E_55_@_V_29"],
    "chest hurts": ["E_53", "E_55_@_V_29"],
    "chest ache": ["E_53", "E_55_@_V_29"],
    "lower chest pain": ["E_53", "E_55_@_V_29"],
    "छातीत दुखत": ["E_53", "E_55_@_V_29"],
    "छातीत दुखत आहे": ["E_53", "E_55_@_V_29"],
    "upper chest pain": ["E_53", "E_55_@_V_101"],
    "side chest pain": ["E_53", "E_55_@_V_55"],
    "stomach pain": ["E_53", "E_55_@_V_187"],
    "abdominal pain": ["E_53", "E_55_@_V_187"],
    "belly pain": ["E_53", "E_55_@_V_187"],
    "pet dard": ["E_53", "E_55_@_V_187"],
    "पोट दुखते": ["E_53", "E_55_@_V_187"],
    "पोट दुखत": ["E_53", "E_55_@_V_187"],
    "epigastric pain": ["E_53", "E_55_@_V_197"],
    "back pain": ["E_53", "E_55_@_V_40"],
    "lower back pain": ["E_53", "E_55_@_V_40"],
    "low back pain": ["E_53", "E_55_@_V_40"],
    "kamar dard": ["E_53", "E_55_@_V_40"],
    "kamar mein dard": ["E_53", "E_55_@_V_40"],
    "कंबर दुखत आहे": ["E_53", "E_55_@_V_40"],
    "कंबर दुखते": ["E_53", "E_55_@_V_40"],
    "lumbar pain": ["E_53", "E_55_@_V_40"],
    "flank pain": ["E_53", "E_55_@_V_84"],
    "kidney pain": ["E_53", "E_55_@_V_113"],
    "pelvic pain": ["E_53", "E_55_@_V_188"],
    "pain while urinating": ["E_53", "E_55_@_V_185"],
    "burning urination": ["E_53", "E_55_@_V_185"],
    "burning while urinating": ["E_53", "E_55_@_V_185"],
    "burning pee": ["E_53", "E_55_@_V_185"],
    "burning while peeing": ["E_53", "E_55_@_V_185"],
    "pain when peeing": ["E_53", "E_55_@_V_185"],
    "painful urination": ["E_53", "E_55_@_V_185"],
    "urine pain": ["E_53", "E_55_@_V_185"],
    "peshab mein jalan": ["E_53", "E_55_@_V_185"],
    "peshab karte waqt jalan": ["E_53", "E_55_@_V_185"],
    "peshab mein dard": ["E_53", "E_55_@_V_185"],
    "laghvi kartana jaljal": ["E_53", "E_55_@_V_185"],
    "लघवी करताना जळजळ": ["E_53", "E_55_@_V_185"],
    "लघवीला जळजळ": ["E_53", "E_55_@_V_185"],
    "cannot urinate": ["E_53", "E_55_@_V_185"],
    "unable to urinate": ["E_53", "E_55_@_V_185"],
    "difficulty urinating": ["E_53", "E_55_@_V_185"],
    "urine takes long time": ["E_53", "E_55_@_V_185"],
    "longer time for urine": ["E_53", "E_55_@_V_185"],
    "takes time to pee": ["E_53", "E_55_@_V_185"],
    "pee takes long time": ["E_53", "E_55_@_V_185"],
    "trouble peeing": ["E_53", "E_55_@_V_185"],
    "pain while breathing": ["E_220"],
    "pain in chest while breathing": ["E_220"],
    "chest pain while breathing": ["E_220"],
    "pain on deep breath": ["E_220"],
    "pain with movement": ["E_216"],
    "pain with cough": ["E_221"],
    "temporary inability to inhale": ["E_168"],
    "temporary inability to breathe": ["E_168"],
    "cannot breathe for short time": ["E_168"],
    "unable to breathe or speak": ["E_168"],
    "drooping eyelid": ["E_147"],
    "eyelid drooping": ["E_147"],
    "hard time opening eyelid": ["E_147"],
    "burning pain": ["E_54_@_V_181"],
    "sharp pain": ["E_54_@_V_192"],
    "cramp pain": ["E_54_@_V_182"],
    "heavy pain": ["E_54_@_V_183"],
    "so much pain": ["E_56_@_8"],
    "too much pain": ["E_56_@_8"],
    "severe pain": ["E_56_@_8"],
    "mild pain": ["E_56_@_2"],
    "moderate pain": ["E_56_@_5"],
    "headache": ["E_53", "E_55_@_V_11"],
    "head pain": ["E_53", "E_55_@_V_11"],
    "severe headache": ["E_53", "E_55_@_V_11", "E_56_@_8"],
    "sar dard": ["E_53", "E_55_@_V_11"],
    "sir dard": ["E_53", "E_55_@_V_11"],
    "dok dukhte": ["E_53", "E_55_@_V_11"],
    "डोके दुखते": ["E_53", "E_55_@_V_11"],
    "dizziness": ["E_81"],
    "dizzy": ["E_81"],
    "chakkar": ["E_81"],
    "चक्कर": ["E_81"],
    "rash": ["E_151"],
    "skin rash": ["E_151"],
    "purळ": ["E_151"],
    "पुरळ": ["E_151"],
    "त्वचेवर पुरळ": ["E_151"],
    "itching": ["E_88"],
    "itchy skin": ["E_88"],
    "khujli": ["E_88"],
    "खाज": ["E_88"],
    "swelling": ["E_207"],
    "face swelling": ["E_207"],
    "joint pain": ["E_53", "E_55_@_V_119"],
    "joints me pain": ["E_53", "E_55_@_V_119"],
    "joints mein pain": ["E_53", "E_55_@_V_119"],
    "sandhi dukhne": ["E_53", "E_55_@_V_119"],
    "सांधे दुखणे": ["E_53", "E_55_@_V_119"],
    "diabetes": ["E_69"],
    "history diabetes": ["E_69"],
    "previous diabetes": ["E_69"],
    "high blood pressure": ["E_104"],
    "hypertension": ["E_104"],
    "bp": ["E_104"],
    "asthma": ["E_124"],
    "history asthma": ["E_124"],
    "heart attack": ["E_105"],
    "angina": ["E_105"],
    "previous heart attack": ["E_105"],
    "high cholesterol": ["E_71"],
    "smoking": ["E_79"],
    "smoker": ["E_79"],
    "smoke cigarettes": ["E_79"],
    "family heart disease": ["E_225"],
    "family cardiac disease": ["E_225"],
    "family cardiovascular disease": ["E_225"],
    "family asthma": ["E_87"],
    "family allergy": ["E_86"],
}


def _load_first_json(paths: list[Path]) -> dict:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def evidence_metadata() -> dict:
    raw = _load_first_json(EVIDENCE_PATHS)
    if not raw:
        return {}
    if "evidences" in raw and isinstance(raw["evidences"], dict):
        return raw["evidences"]
    return raw


@lru_cache(maxsize=1)
def condition_metadata() -> dict:
    raw = _load_first_json(CONDITION_PATHS)
    if not raw:
        return {}
    if "conditions" in raw and isinstance(raw["conditions"], dict):
        return raw["conditions"]
    return raw


def split_evidence_code(code: Any) -> tuple[str, str | None]:
    text = str(code or "").strip().strip("'\"").upper()
    if "_@_" not in text:
        return text, None
    base, value = text.split("_@_", 1)
    return base, value


def _entry_text(entry: Any, keys: tuple[str, ...], fallback: str) -> str:
    if not isinstance(entry, dict):
        return fallback
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _value_entry_text(value_entry: Any, fallback: str) -> str:
    if isinstance(value_entry, dict):
        return _entry_text(
            value_entry,
            ("en", "name", "label", "text", "value", "description"),
            fallback,
        )
    if isinstance(value_entry, str) and value_entry.strip():
        return value_entry.strip()
    return fallback


def _canonical_value_key(value_code: str | None) -> str | None:
    if not value_code:
        return None
    value = str(value_code).strip().upper()
    if value and value.isdigit():
        return value
    return value


def _value_text(entry: Any, value_code: str | None) -> str | None:
    if not value_code or not isinstance(entry, dict):
        return None
    value_code = _canonical_value_key(value_code)
    value_meaning = entry.get("value_meaning") or entry.get("value-meaning") or {}
    if isinstance(value_meaning, dict):
        value_entry = (
            value_meaning.get(value_code)
            or value_meaning.get(str(value_code).upper())
            or value_meaning.get(str(value_code).lower())
        )
        if value_entry is not None:
            return _value_entry_text(value_entry, str(value_code))

    values = (
        entry.get("values")
        or entry.get("possible-values")
        or entry.get("possible_values")
        or entry.get("choices")
        or {}
    )
    if isinstance(values, dict):
        value_entry = values.get(value_code) or values.get(value_code.upper()) or values.get(value_code.lower())
        if isinstance(value_entry, dict):
            return _entry_text(value_entry, ("name", "label", "text", "value", "description"), value_code)
        if isinstance(value_entry, str):
            return value_entry
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, dict):
                continue
            item_code = str(item.get("code") or item.get("id") or item.get("value") or "").upper()
            if item_code == value_code.upper():
                return _entry_text(item, ("name", "label", "text", "description"), value_code)
    if isinstance(values, list) and str(value_code) in {str(item).upper() for item in values}:
        return str(value_code)
    return None


def _question_text(entry: Any, fallback: str) -> str:
    return _entry_text(
        entry,
        ("question_en", "question", "prompt", "description", "display_name", "label", "name"),
        fallback,
    )


def _display_name(entry: Any, base_code: str, question: str) -> str:
    name = _entry_text(entry, ("display_name", "condition_name", "label", "name"), "")
    if not name or CODE_LIKE_NAME_RE.fullmatch(name) or name.upper() == base_code.upper():
        return question
    return name


def decode_evidence(code: Any) -> dict:
    """Decode one evidence code into display fields."""
    original = str(code or "").strip().strip("'\"").upper()
    base_code, value_code = split_evidence_code(original)
    entry = evidence_metadata().get(base_code, {})
    question = _question_text(entry, f"Unknown evidence {base_code}")
    name = _display_name(entry, base_code, question)
    value = _value_text(entry, value_code)

    meaning = question
    if value:
        meaning = f"{question} {value}"

    return {
        "code": original,
        "base_code": base_code,
        "value_code": value_code,
        "name": name,
        "question": question,
        "value": value,
        "meaning": meaning,
        "known": bool(entry),
    }


def decode_evidences(codes: list[str] | tuple[str, ...] | str) -> list[dict]:
    if isinstance(codes, str):
        found = EVIDENCE_CODE_RE.findall(codes)
    else:
        found = list(codes or [])
    return [decode_evidence(code) for code in found]


def condition_severity(condition: Any) -> int | None:
    """
    Return the official DDXPlus severity rating for a pathology, or None if
    the condition is unknown or the loaded release_conditions.json has no
    severity field (e.g. the bundled demo subset before it's replaced).

    DDXPlus rates severity on a 1-5 scale per condition. This project does
    not hardcode which end of that scale means "more severe" — see
    utils.severity_engine.describe_severity_scale() to confirm the polarity
    against your own copy of release_conditions.json before trusting any
    severity-based triage logic.
    """
    decoded = decode_condition(condition)
    if not decoded.get("known"):
        return None
    entry = condition_metadata().get(decoded.get("metadata_key", ""), {})
    severity = entry.get("severity") if isinstance(entry, dict) else None
    try:
        return int(severity) if severity is not None else None
    except (TypeError, ValueError):
        return None


def decode_condition(condition: Any) -> dict:
    text = str(condition or "").strip()
    if not text:
        return {"condition": "", "display_name": "Unknown condition", "known": False, "severity": None}

    lower = text.lower()
    for key, entry in condition_metadata().items():
        if not isinstance(entry, dict):
            continue
        names = [
            str(key),
            str(entry.get("cond-name-eng", "")),
            str(entry.get("cond_name_eng", "")),
            str(entry.get("cond-name-fr", "")),
            str(entry.get("condition_name", "")),
            str(entry.get("display_name", "")),
            str(entry.get("name", "")),
            *[str(alias) for alias in entry.get("aliases", []) if alias],
        ]
        if any(name.lower() == lower for name in names if name):
            display = _entry_text(
                entry,
                ("display_name", "condition_name", "cond-name-eng", "cond_name_eng", "name"),
                str(key),
            )
            severity = entry.get("severity")
            try:
                severity = int(severity) if severity is not None else None
            except (TypeError, ValueError):
                severity = None
            return {
                "condition": text,
                "display_name": display,
                "known": True,
                "metadata_key": key,
                "severity": severity,
            }
    return {"condition": text, "display_name": text, "known": False, "severity": None}


def evidence_markdown(
    codes: list[str],
    initial_evidence: str | None = None,
    limit: int = 40,
    show_codes: bool = False,
) -> str:
    rows = decode_evidences(codes)
    initial = str(initial_evidence or "").strip().upper()
    if not rows and not initial:
        return "## 🧠 Decoded Evidence Meaning\n\nNo matching symptom meanings were detected."

    lines = ["## 🧠 Decoded Evidence Meaning\n"]
    if initial:
        decoded_initial = decode_evidence(initial)
        if show_codes:
            lines.append(f"**Initial evidence:** `{decoded_initial['code']}` - {decoded_initial['meaning']}\n")
        else:
            lines.append(f"**Initial symptom understood:** {decoded_initial['meaning']}\n")

    if show_codes:
        lines.append("| Code | Meaning |")
        lines.append("|------|---------|")
        for row in rows[:limit]:
            marker = " *(initial)*" if initial and row["code"] == initial else ""
            lines.append(f"| `{row['code']}` | {row['meaning']}{marker} |")
        if len(rows) > limit:
            lines.append(f"| ... | {len(rows) - limit} more evidence items hidden |")
        return "\n".join(lines)

    lines.append("| # | Evidence meaning |")
    lines.append("|---|------------------|")
    for idx, row in enumerate(rows[:limit], 1):
        marker = " *(initial)*" if initial and row["code"] == initial else ""
        lines.append(f"| {idx} | {row['meaning']}{marker} |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} more evidence items hidden |")
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f ]+", " ", str(text or "").lower())


def _normalized_curated_aliases() -> set[str]:
    return {_normalize_text(alias).strip() for alias in CURATED_TEXT_ALIASES}


def _locate_alias(normalized_text: str, alias: str) -> tuple[int, int] | None:
    """Find the (start, end) character span of `alias` as a whole phrase."""
    pattern = r"(?<!\S)" + re.escape(alias) + r"(?!\S)"
    match = re.search(pattern, normalized_text)
    return match.span() if match else None


def _should_try_semantic_fallback(text: str) -> bool:
    normalized = _normalize_text(text)
    tokens = set(normalized.split())
    if not tokens:
        return False
    if QUESTION_ONLY_RE.search(str(text or "")) and not (tokens & SEMANTIC_SYMPTOM_HINTS):
        return False
    if len(tokens) <= 2 and not (tokens & SEMANTIC_SYMPTOM_HINTS):
        return False
    return bool(tokens & SEMANTIC_SYMPTOM_HINTS)


def _add_alias(alias_map: dict[str, list[str]], alias: Any, codes: Any) -> None:
    normalized = _normalize_text(str(alias)).strip()
    if len(normalized) < 3:
        return
    if isinstance(codes, str):
        candidate_codes = [codes]
    else:
        candidate_codes = list(codes or [])
    valid_codes = []
    for code in candidate_codes:
        base_code, _ = split_evidence_code(code)
        if base_code in evidence_metadata():
            valid_codes.append(str(code).upper())
    if not valid_codes:
        return
    existing = alias_map.setdefault(normalized, [])
    for code in valid_codes:
        if code not in existing:
            existing.append(code)


def _question_aliases(question: str) -> list[str]:
    normalized = _normalize_text(question).strip()
    aliases = {normalized}
    replacements = (
        "do you have ",
        "do you feel ",
        "are you experiencing ",
        "are you feeling ",
        "have you had ",
        "have you noticed ",
        "does the person have ",
    )
    for prefix in replacements:
        if normalized.startswith(prefix):
            aliases.add(normalized[len(prefix):].strip())
    aliases = {alias.strip() for alias in aliases if len(alias.strip()) >= 3}
    return sorted(aliases, key=len, reverse=True)


def _value_aliases(value_text: str) -> list[str]:
    normalized = _normalize_text(value_text).strip()
    aliases = {normalized}
    without_side = re.sub(r"\b[rl]\b", " ", normalized).strip()
    without_side = re.sub(r"\s+", " ", without_side)
    if without_side:
        aliases.add(without_side)
    return sorted(alias for alias in aliases if len(alias) >= 3)


def _is_location_value_code(code: str) -> bool:
    base, value = split_evidence_code(code)
    return base == "E_55" and bool(value)


def _is_location_only_match(alias: str, codes: list[str]) -> bool:
    if not codes:
        return False
    if alias in _normalized_curated_aliases():
        return False
    return all(_is_location_value_code(code) for code in codes)


def _has_local_pain_context(normalized_text: str, span: tuple[int, int]) -> bool:
    start, end = span
    window = normalized_text[max(0, start - 36): min(len(normalized_text), end + 36)]
    return bool(PAIN_CONTEXT_RE.search(window))


def _evidence_strength(alias: str, codes: list[str]) -> str:
    if alias in _normalized_curated_aliases():
        return "curated_alias"
    if all(_is_location_value_code(code) for code in codes):
        return "location_with_context"
    return "metadata_alias"


def _is_generic_initial_code(code: str) -> bool:
    base, value = split_evidence_code(code)
    return base in GENERIC_INITIAL_EVIDENCE_CODES and not value


def select_initial_evidence(items: list[Any]) -> str:
    """
    Pick the best initial evidence from inferred matches or raw codes.

    The DDXPlus model treats INITIAL_EVIDENCE as the chief complaint. For
    plain text inference, generic parent findings such as E_53 ("pain
    somewhere") often appear before the specific value code. Prefer the first
    specific code so "burning urination" becomes urethral pain rather than
    generic pain.
    """
    codes = []
    for item in items or []:
        code = item.get("code") if isinstance(item, dict) else item
        code = str(code or "").strip().upper()
        if code and code not in codes:
            codes.append(code)

    for code in codes:
        if not _is_generic_initial_code(code):
            return code
    return codes[0] if codes else ""


def alias_to_evidence_map() -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}
    for code, entry in evidence_metadata().items():
        if not isinstance(entry, dict):
            continue
        aliases = entry.get("aliases", []) or []
        question = _question_text(entry, "")
        name = _display_name(entry, str(code), question)
        for candidate in [name, question, *aliases, *_question_aliases(question)]:
            _add_alias(alias_map, candidate, code)

        value_meaning = entry.get("value_meaning") or entry.get("value-meaning") or {}
        if code == "E_55" and isinstance(value_meaning, dict):
            for value_code, value_entry in value_meaning.items():
                value_text = _value_entry_text(value_entry, "")
                if not value_text or value_text.lower() in {"na", "none", "nowhere"}:
                    continue
                coded_value = f"{code}_@_{str(value_code).upper()}"
                for alias in _value_aliases(value_text):
                    _add_alias(alias_map, alias, coded_value)
                    _add_alias(alias_map, f"{alias} pain", ["E_53", coded_value])
                    _add_alias(alias_map, f"pain in {alias}", ["E_53", coded_value])

    for alias, codes in CURATED_TEXT_ALIASES.items():
        _add_alias(alias_map, alias, codes)
    return alias_map


def infer_evidence_codes_from_text(text: str, include_denied: bool = False) -> list[dict]:
    """
    Best-effort demo mapper from human symptoms to evidence codes.

    This is not a clinical parser. It exists so the chatbot/demo can bridge
    plain symptom text to the structured evidence-code model. If enabled,
    BioBERT is used only as a semantic fallback after exact alias matching.

    Negation handling: an exact alias match whose position falls inside a
    negated scope ("no chest pain", "chest pain ruled out", "chest pain
    nahi hai" — see utils.negation) is treated as a *denied* finding, not a
    positive one, and is excluded from the returned list unless
    `include_denied=True`, in which case denied items are appended at the
    end with `"negated": True`.

    Known limitation: negation filtering currently covers exact alias
    matches only. The BioBERT semantic fallback (used when no exact alias
    matches at all) does not yet carry per-match position information back
    from `semantic_alias_matches`, so it is not negation-filtered. In
    practice this only matters when a user's *entire* message has no exact
    alias hit, which is the narrower case.
    """
    normalized = _normalize_text(text)
    matches = []
    denied_matches = []
    seen = set()
    seen_denied = set()
    alias_map = alias_to_evidence_map()
    for alias, codes in sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True):
        if not alias:
            continue
        span = _locate_alias(normalized, alias)
        if span is None:
            continue
        if _is_location_only_match(alias, codes) and not _has_local_pain_context(normalized, span):
            continue
        negated = is_negated(normalized, span[0], span[1])
        strength = _evidence_strength(alias, codes)
        for code in codes:
            if negated:
                if code in seen or code in seen_denied:
                    continue
                seen_denied.add(code)
                decoded = decode_evidence(code)
                denied_matches.append({
                    "source_text": alias,
                    "code": code,
                    "meaning": decoded["meaning"],
                    "match_type": "exact_alias",
                    "evidence_strength": strength,
                    "negated": True,
                })
                continue
            if code in seen:
                continue
            seen.add(code)
            decoded = decode_evidence(code)
            matches.append({
                "source_text": alias,
                "code": code,
                "meaning": decoded["meaning"],
                "match_type": "exact_alias",
                "evidence_strength": strength,
                "negated": False,
            })

    def _finish(extra_matches: list[dict] | None = None) -> list[dict]:
        result = matches + (extra_matches or [])
        return result + denied_matches if include_denied else result

    if matches:
        return _finish()

    if not _should_try_semantic_fallback(text):
        return _finish()

    try:
        from utils.biobert_embedder import semantic_alias_matches
    except Exception:
        return _finish()

    semantic_matches = []
    for semantic_match in semantic_alias_matches(text, alias_map, exclude_codes=seen | seen_denied):
        for code in semantic_match["codes"]:
            if code in seen:
                continue
            seen.add(code)
            decoded = decode_evidence(code)
            semantic_matches.append({
                "source_text": semantic_match["alias"],
                "code": code,
                "meaning": decoded["meaning"],
                "match_type": "biobert_semantic",
                "evidence_strength": "semantic_fallback",
                "similarity": semantic_match["similarity"],
                "model": semantic_match["model"],
                "negated": False,
            })
    return _finish(semantic_matches)


def explicit_evidence_codes(text: str) -> list[str]:
    """Return evidence codes typed directly by the user."""
    return list(dict.fromkeys(code.upper() for code in EVIDENCE_CODE_RE.findall(str(text or ""))))


def resolve_evidence_input(text: str) -> tuple[list[str], list[dict]]:
    """
    Accept either raw evidence codes, human text, or a mixture of both.

    Returns:
        (codes, inferred_matches)
    """
    direct_codes = explicit_evidence_codes(text)
    inferred = infer_evidence_codes_from_text(text)
    codes = list(dict.fromkeys([*direct_codes, *[item["code"] for item in inferred]]))
    return codes, inferred


def resolve_initial_evidence_input(initial_text: str, fallback_codes: list[str] | None = None) -> tuple[str, dict | None]:
    """
    Accept an initial evidence code or a human phrase such as "cough".

    If no initial evidence can be resolved, fall back to the first evidence code.
    """
    direct_codes = explicit_evidence_codes(initial_text)
    if direct_codes:
        return direct_codes[0], None

    inferred = infer_evidence_codes_from_text(initial_text)
    if inferred:
        return select_initial_evidence(inferred), inferred[0]

    fallback = list(fallback_codes or [])
    if fallback:
        return select_initial_evidence(fallback), None
    return "", None


def parse_age_sex_from_text(text: str) -> tuple[int | None, str]:
    raw = str(text or "")
    age = None
    age_patterns = [
        r"\bage\s*[:=]?\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:years?|yrs?|yo)\s*old\b",
        r"\b(\d{1,3})\s*(?:years?|yrs?|yo)\b",
        r"\b(?:i\s*am|im|i'm)\s*(\d{1,3})\b(?!\s*(?:am|pm))",
        r"\b(\d{1,3})\s*[- ]?(?:year|yr)[- ]?old\b",
    ]
    for pattern in age_patterns:
        age_match = re.search(pattern, raw, flags=re.I)
        if not age_match:
            continue
        try:
            candidate = int(age_match.group(1))
        except ValueError:
            continue
        if 0 <= candidate <= 110:
            age = candidate
            break

    lowered = raw.lower()
    if re.search(r"\b(female|woman|girl|lady|sex\s*[:=]?\s*f)\b", lowered):
        sex = "F"
    elif re.search(r"\b(male|man|boy|sex\s*[:=]?\s*m)\b", lowered):
        sex = "M"
    else:
        sex = "U"
    return age, sex
