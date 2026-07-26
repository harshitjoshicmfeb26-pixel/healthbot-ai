"""
utils/ollama_client.py
──────────────────────
Optional local Ollama response formatter.

Qwen/Ollama is not used as the diagnosis engine. It only rewrites structured
model and safety output into a safe, concise, multilingual response.

The system prompt below *asks* the model not to invent diagnoses or give
dosages. `verify_grounded_response()` actually *checks* the output against
that rule before it's allowed to reach the user — see its docstring. If the
check fails, `format_with_ollama` returns "" and the caller falls back to
`utils.response_summarizer.summarize_response` (the deterministic template),
which cannot hallucinate a disease because it only renders fields already
present in the structured payload.
"""

import json
import re
from typing import Dict

try:
    import requests
except Exception:  # pragma: no cover - requests should be installed via requirements
    requests = None

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_ENABLED,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    USE_OLLAMA_RESPONSE_FORMATTER,
)


SYSTEM_PROMPT = """You are a healthcare response formatter, not a doctor.
You must not diagnose.
You must not override the ML classifier predictions.
You must only summarize the provided structured data.
You must not add diseases that are not present in predictions.
You must not recommend medicines or dosage.
You must include disclaimer.
If red flags are present, prioritize urgent medical care.
Answer in the same language style as the user: English, Hinglish, or simple Marathi.
Keep the response concise and safe.

Few-shot examples:

Example 1:
Input symptoms: fever and headache
Prediction data: Common Cold 45%, Migraine 20%
Response: Your symptoms were normalized as fever and headache. The ML model suggests possible conditions such as Common Cold and Migraine, but this is not a diagnosis. Rest, monitor symptoms, and consult a qualified clinician if symptoms worsen. Disclaimer: This tool does not replace medical advice.

Example 2:
Input symptoms: mujhe bukhar aur sar dard hai
Prediction data: Common Cold 42%, Migraine 21%
Response: Aapke symptoms fever aur headache ke roop me normalize hue hain. ML model kuch possible conditions dikhata hai, jaise Common Cold aur Migraine. Yeh diagnosis nahi hai. Agar symptoms badh rahe hain, doctor se consult karein. Disclaimer: This tool medical advice ka replacement nahi hai.

Example 3:
Input symptoms: मला ताप आणि डोकेदुखी आहे
Prediction data: Common Cold 40%, Migraine 18%
Response: तुमची लक्षणे fever आणि headache अशी normalize झाली आहेत. ML model काही संभाव्य स्थिती दाखवतो, जसे Common Cold आणि Migraine. हे निदान नाही. लक्षणे वाढल्यास डॉक्टरांचा सल्ला घ्या. Disclaimer: हे साधन वैद्यकीय सल्ल्याचा पर्याय नाही.
"""


def _enabled() -> bool:
    return bool(OLLAMA_ENABLED and USE_OLLAMA_RESPONSE_FORMATTER)


_DOSAGE_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|micrograms?|milligrams?|ml|millilit(?:er|re)s?|iu|units?)\b"
    r"|\b\d+\s*(tablets?|capsules?|pills?|doses?)\b"
    r"|\btimes?\s+a\s+day\b|\btimes?\s+daily\b",
    re.IGNORECASE,
)


def _mentions_dosage(text: str) -> bool:
    return bool(_DOSAGE_RE.search(text or ""))


def _mentions_ungrounded_disease(text: str, allowed_diseases: set[str]) -> str | None:
    """Return the first known disease name mentioned that ISN'T in `allowed_diseases`."""
    try:
        from model.predictor import known_pathology_names
        all_known = known_pathology_names()
    except Exception:
        # If the model/classes can't be loaded for some reason, this check
        # is skipped rather than blocking the whole response on a missing
        # signal — the dosage check above still applies regardless.
        return None

    lowered_text = (text or "").lower()
    allowed_lower = {d.lower() for d in allowed_diseases}
    for name in all_known:
        if not name or name.lower() in allowed_lower:
            continue
        if re.search(re.escape(name.lower()), lowered_text):
            return name
    return None


def verify_grounded_response(text: str, payload: Dict) -> tuple[bool, str]:
    """
    Guard the LLM-generated explanation before it reaches the user.

    The system prompt already *asks* the model not to hallucinate diseases
    or give dosages — this function actually *checks* the output:
      1. Rejects responses containing a dosage/medication-amount pattern.
      2. Rejects responses naming a pathology that is in the model's known
         49-class vocabulary but NOT in this case's current top-k
         predictions (i.e. the model invented a diagnosis not supported by
         the structured classifier output it was given).

    Returns (is_valid, reason); `reason` is empty when valid.
    """
    if not text or not text.strip():
        return False, "empty response"
    if _mentions_dosage(text):
        return False, "response mentions a dosage/medication amount"

    predictions = payload.get("predictions") or []
    allowed = {str(item.get("disease", "")) for item in predictions if item.get("disease")}
    allowed |= {str(item.get("pathology", "")) for item in predictions if item.get("pathology")}

    ungrounded = _mentions_ungrounded_disease(text, allowed)
    if ungrounded:
        return False, f"response names '{ungrounded}', which is outside the current candidate list"

    return True, ""


def format_with_ollama(payload: Dict) -> str:
    """
    Format structured prediction output with local Ollama.

    Returns an empty string if Ollama is disabled or unavailable. Callers should
    fall back to `utils.response_summarizer.summarize_response`.
    """
    if not _enabled():
        return ""
    if requests is None:
        return ""

    prompt = (
        SYSTEM_PROMPT
        + "\n\nStructured data to summarize. Do not invent facts:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.8,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        generated = str(data.get("response", "")).strip()
    except Exception:
        return ""

    is_valid, reason = verify_grounded_response(generated, payload)
    if not is_valid:
        print(f"[ollama_client] Discarding ungrounded response ({reason}); falling back to template.")
        return ""
    return generated
