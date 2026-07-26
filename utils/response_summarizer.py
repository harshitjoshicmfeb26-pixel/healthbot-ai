"""
utils/response_summarizer.py
────────────────────────────
Grounded deterministic healthcare response formatter.

This is the fallback when Ollama is disabled or unavailable. It never diagnoses,
does not recommend medicines, and only summarizes facts already present in the
structured pipeline payload.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List


def _language_style(detected_language) -> str:
    if isinstance(detected_language, dict):
        language = detected_language.get("language", "english")
    else:
        language = detected_language or "english"

    if language == "marathi_devanagari":
        return "marathi"
    if language in {"hinglish", "romanized_marathi", "mixed"}:
        return "hinglish"
    return "english"


def _variant(key: str, options: list[str]) -> str:
    if not options:
        return ""
    digest = hashlib.sha1(str(key or "").encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _prediction_lines(predictions: List[Dict], limit: int = 3) -> List[str]:
    if not predictions:
        return ["- No model prediction was available."]

    lines = []
    for pred in predictions[:limit]:
        disease = pred.get("disease") or pred.get("pathology") or "Unknown condition"
        confidence = pred.get("confidence_pct", "n/a")
        flag = pred.get("flag", "")
        score_type = pred.get("score_type", "model confidence")
        lines.append(f"- {disease} ({score_type}: {confidence}, {flag})")
    return lines


def _symptom_text(normalized_symptoms) -> str:
    if isinstance(normalized_symptoms, list):
        return ", ".join(str(item) for item in normalized_symptoms if str(item).strip()) or "not clearly identified"
    return str(normalized_symptoms or "not clearly identified")


def _evidence_meanings(evidence_bridge: Dict, limit: int = 6) -> list[str]:
    inferred = evidence_bridge.get("inferred_evidences") or []
    meanings = []
    for item in inferred[:limit]:
        meaning = str(item.get("meaning") or "").strip()
        if meaning and meaning not in meanings:
            meanings.append(meaning)
    return meanings


def _bridge_status(evidence_bridge: Dict) -> str:
    if evidence_bridge.get("inferred_evidences"):
        match_types = {
            str(item.get("match_type", "unknown"))
            for item in evidence_bridge.get("inferred_evidences", [])
        }
        if "biobert_semantic" in match_types:
            return "semantic evidence match"
        if "exact_alias" in match_types:
            return "exact evidence match"
        return "evidence match"
    return str(evidence_bridge.get("mode") or "no evidence match")


def _fallback_guidance(style: str, symptoms: str, evidence_bridge: Dict) -> str:
    mode = evidence_bridge.get("mode", "free_text_fallback")
    bridge_message = str(evidence_bridge.get("message") or "").strip()
    if not bridge_message:
        quality = evidence_bridge.get("evidence_quality") or {}
        bridge_message = str(quality.get("message") or "").strip()
    if not bridge_message:
        scope = evidence_bridge.get("scope_warning") or {}
        bridge_message = str(scope.get("message") or "").strip()

    if mode in {"weak_evidence", "model_scope_limited"}:
        status = bridge_message or "The classifier was not run for this message."
        if style == "marathi":
            return (
                "### मॉडेल स्थिती:\n"
                f"{status}\n\n"
                "### पुढे काय लिहावे:\n"
                "कृपया मुख्य लक्षण, जागा, duration, severity, आणि वय/लिंग अधिक स्पष्ट द्या."
            )
        if style == "hinglish":
            return (
                "### Model Status:\n"
                f"{status}\n\n"
                "### Better input:\n"
                "Main symptom, location, duration, severity, age/sex ko direct words mein likhein."
            )
        return (
            "### Model Status:\n"
            f"{status}\n\n"
            "### Better input:\n"
            "Describe the main symptom directly, including location, duration, severity, age, and sex if known."
        )

    if style == "marathi":
        return (
            "### मॅपिंग स्थिती:\n"
            f"मी `{symptoms}` हे अधिकृत DDXPlus symptom evidence मध्ये विश्वासाने मॅप करू शकलो नाही. "
            f"Pipeline mode: `{mode}`.\n\n"
            "### पुढे काय लिहावे:\n"
            "कृपया लक्षणे अधिक थेट शब्दांत लिहा, उदा. `chest pain`, `burning urination`, "
            "`cough with fever`, `lower back pain`, आणि शक्य असल्यास duration/severity द्या."
        )
    if style == "hinglish":
        return (
            "### Mapping Status:\n"
            f"Main symptom text `{symptoms}` ko official DDXPlus evidence category se confidently map nahi kar paya. "
            f"Pipeline mode: `{mode}`.\n\n"
            "### Better input:\n"
            "Symptoms ko direct words mein likhein, jaise `chest pain`, `burning urination`, "
            "`cough with fever`, `lower back pain`, plus duration/severity if possible."
        )
    return (
        "### Mapping Status:\n"
        f"I could not confidently map `{symptoms}` to an official DDXPlus evidence category. "
        f"Pipeline mode: `{mode}`.\n\n"
        "### Better input:\n"
        "Try direct symptom wording such as `chest pain`, `burning urination`, "
        "`cough with fever`, or `lower back pain`, and add duration/severity if known."
    )


def _confidence_sentence(style: str, confidence_status: str, top: Dict, key: str) -> str:
    disease = top.get("disease", "Unknown condition")
    confidence = top.get("confidence_pct", "n/a")
    if style == "marathi":
        variants = {
            "high": [
                f"सध्याच्या evidence नुसार strongest model match **{disease}** आहे ({confidence}).",
                f"मॉडेलने सर्वाधिक जुळणारी स्थिती **{disease}** दाखवली आहे ({confidence}).",
            ],
            "medium": [
                f"मॉडेलला **{disease}** हा जवळचा match वाटतो, पण confidence मध्यम आहे ({confidence}).",
                f"सध्याच्या माहितीवर **{disease}** top match आहे; confidence {confidence} आहे.",
            ],
            "low": [
                f"Top match **{disease}** आहे, पण confidence कमी आहे ({confidence}); हे rough signal समजा.",
                f"मॉडेलने **{disease}** दाखवले, पण confidence कमी असल्याने अजून माहिती उपयोगी ठरेल.",
            ],
        }
    elif style == "hinglish":
        variants = {
            "high": [
                f"Given mapped evidence, strongest model match **{disease}** hai ({confidence}).",
                f"Current structured evidence ke hisaab se top match **{disease}** hai ({confidence}).",
            ],
            "medium": [
                f"Closest match **{disease}** hai, lekin confidence medium hai ({confidence}).",
                f"Model **{disease}** ko top match dikha raha hai; confidence {confidence} hai.",
            ],
            "low": [
                f"Top match **{disease}** hai, but confidence low hai ({confidence}); isko rough signal samjhein.",
                f"Model ne **{disease}** suggest kiya, lekin low confidence ka matlab details insufficient ho sakti hain.",
            ],
        }
    else:
        variants = {
            "high": [
                f"From the mapped evidence, the strongest model match is **{disease}** ({confidence}).",
                f"The structured classifier currently ranks **{disease}** highest ({confidence}).",
            ],
            "medium": [
                f"The closest model match is **{disease}**, with medium confidence ({confidence}).",
                f"The model ranks **{disease}** first, but confidence is only medium ({confidence}).",
            ],
            "low": [
                f"The top model match is **{disease}**, but confidence is low ({confidence}); treat it as a weak signal.",
                f"The classifier returned **{disease}**, but low confidence means the evidence is not very discriminative yet.",
            ],
        }
    return _variant(key + confidence_status, variants.get(confidence_status, variants["low"]))


def summarize_response(payload: Dict) -> str:
    """Build a safe response from structured prediction and safety data."""
    style = _language_style(payload.get("detected_language"))
    symptoms = _symptom_text(payload.get("normalized_symptoms"))
    predictions = payload.get("predictions", []) or []
    red_flag = payload.get("red_flag_result", {}) or {}
    missing_info = payload.get("missing_info", []) or []
    evidence_bridge = payload.get("evidence_bridge", {}) or {}
    confidence_status = payload.get("confidence_status", "unknown")
    original_text = str(payload.get("original_text") or symptoms)
    key = original_text + str(predictions[:2]) + str(red_flag)

    evidence_meanings = _evidence_meanings(evidence_bridge)
    has_usable_prediction = bool(predictions) and predictions[0].get("flag") not in {"INFO", "ERROR"}
    has_evidence = bool(evidence_meanings)
    pred_lines = "\n".join(_prediction_lines(predictions))
    red_message = red_flag.get("safety_message", "No red-flag warning was detected.")

    if not has_evidence or not has_usable_prediction:
        fallback = _fallback_guidance(style, symptoms, evidence_bridge)
        safety = f"\n\n### Safety:\n{red_message}" if red_flag else ""
        disclaimer = (
            "\n\n### Disclaimer:\n"
            "This tool is for educational information only and does not replace professional medical advice."
        )
        return fallback + safety + disclaimer

    bridge_line = "; ".join(evidence_meanings)
    top_sentence = _confidence_sentence(style, confidence_status, predictions[0], key)
    missing = ", ".join(missing_info) if missing_info else ""
    bridge_status = _bridge_status(evidence_bridge)

    if style == "marathi":
        next_step = (
            f"Missing/extra info: {missing}. " if missing else ""
        ) + "लक्षणे वाढल्यास, severe वाटल्यास किंवा red-flag असेल तर qualified clinician शी संपर्क करा."
        return (
            "### लक्षणांचा सारांश:\n"
            f"{symptoms}\n\n"
            "### Evidence Used By The Model:\n"
            f"{bridge_line}\n"
            f"Match type: {bridge_status}\n\n"
            "### Model Output:\n"
            f"{top_sentence}\n"
            f"{pred_lines}\n\n"
            "### Safety:\n"
            f"{red_message}\n\n"
            "### Next Step:\n"
            f"{next_step}\n\n"
            "### Disclaimer:\n"
            "हे साधन शैक्षणिक माहितीकरिता आहे. हे वैद्यकीय निदान किंवा उपचारांचा पर्याय नाही."
        )

    if style == "hinglish":
        next_step = (
            f"Missing/extra info: {missing}. " if missing else ""
        ) + "Agar symptoms severe, worsening, ya red-flag type hain, clinician se consult karein."
        return (
            "### Summary:\n"
            f"Normalized symptoms: {symptoms}\n\n"
            "### Evidence Used By The Model:\n"
            f"{bridge_line}\n"
            f"Match type: {bridge_status}\n\n"
            "### Model Output:\n"
            f"{top_sentence}\n"
            f"{pred_lines}\n\n"
            "### Safety:\n"
            f"{red_message}\n\n"
            "### Next Step:\n"
            f"{next_step}\n\n"
            "### Disclaimer:\n"
            "This tool is for educational information only and does not replace professional medical advice."
        )

    next_step = (
        f"Missing/extra information: {missing}. " if missing else ""
    ) + "If symptoms are severe, worsening, or match a red flag, contact a qualified clinician."
    return (
        "### Symptom Summary:\n"
        f"Normalized symptoms: {symptoms}\n\n"
        "### Evidence Used By The Model:\n"
        f"{bridge_line}\n"
        f"Match type: {bridge_status}\n\n"
        "### Model Output:\n"
        f"{top_sentence}\n"
        f"{pred_lines}\n\n"
        "### Urgency/Safety:\n"
        f"{red_message}\n\n"
        "### What You Should Do:\n"
        f"{next_step}\n\n"
        "### Disclaimer:\n"
        "This tool is for educational information only and does not replace professional medical advice, diagnosis, or treatment."
    )
