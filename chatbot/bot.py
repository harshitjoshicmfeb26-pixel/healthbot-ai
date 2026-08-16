"""
chatbot/bot.py
──────────────
Rule-aware multilingual symptom-checker chatbot.

New flow:
Greeting
→ detect language
→ normalize symptoms
→ check red flags
→ collect a few missing clinical slots
→ predict disease/category with supervised ML
→ format with local Ollama if enabled
→ fallback to deterministic template summarizer

Important:
Qwen/Ollama is not used as a diagnosis engine. The ML classifier produces the
prediction data. Red-flag triage is rule-based. Ollama only formats structured
output into safe language.
"""

import re
from typing import Dict, List, Tuple

from config import CONFIDENCE_THRESHOLD, MAX_CLARIFICATION_TURNS, RED_FLAG_FIRST
from model.predictor import explain_case, predict, predict_case, semantic_search, semantic_search_case
from utils.ddxplus_decoder import infer_evidence_codes_from_text, parse_age_sex_from_text, select_initial_evidence
from utils.multilingual_normalizer import normalize_symptoms
from utils.ollama_client import format_with_ollama
from utils.biobert_embedder import biobert_status
from utils.red_flag_rules import detect_red_flags
from utils.response_summarizer import summarize_response
from utils.severity_engine import evaluate_differential_severity


STATE_GREETING = "greeting"
STATE_COLLECTING = "collecting"
STATE_DONE = "done"


_GREETING = (
    "Hello! I am **HealthBot**, a local non-RAG healthcare assistant.\n\n"
    "You can describe symptoms in English, Hinglish, Marathi, or Romanized "
    "Marathi. I can help normalize symptoms, check red flags, and show ML model "
    "predictions.\n\n"
    "This is not a medical diagnosis. Type `reset` anytime to start over."
)

_RESET_TRIGGERS = {"reset", "clear", "start over", "restart", "new", "exit", "quit"}
_GREETING_WORDS = {"hi", "hello", "hey", "namaste", "नमस्कार"}
_THANKS_WORDS = {"thanks", "thank you", "thx", "ok thanks", "okay thanks"}
_HELP_HINT = (
    "You can type your symptoms in one sentence, for example: "
    "`I have burning urination for 2 days with lower back pain`."
)
_QUESTION_PREFIXES = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "can",
    "could",
    "should",
    "do",
    "does",
    "did",
    "is",
    "are",
    "will",
    "would",
    "which",
    "who",
    "explain",
)
_SLOT_HINTS = {
    "main_symptoms": "Please describe the main symptoms in one sentence, such as `burning urination with lower back pain`.",
    "age_years": "You can answer with a number, such as `45`, or say `45 years old`.",
    "sex": "You can answer `male`, `female`, `M`, or `F`.",
    "duration": "You can answer like `2 days`, `since yesterday`, or `for 1 week`.",
    "severity": "You can answer like `mild`, `moderate`, `severe`, or a pain score such as `8/10`.",
    "age_group": "You can answer like `child`, `adult`, `elderly`, or give the age such as `45 years old`.",
    "pain_location": "You can answer like `chest`, `left side chest`, `head`, `throat`, or `none`.",
    "previous_disease": "Mention conditions such as `diabetes`, `asthma`, `high BP`, `heart attack`, or say `none`.",
    "family_history": "Mention family history such as `family heart disease`, `family asthma`, or say `none`.",
}
REQUIRED_DETAIL_SLOTS = [
    "age_years",
    "sex",
    "severity",
    "pain_location",
    "duration",
    "previous_disease",
    "family_history",
]
NONE_VALUES = {"none", "no", "nothing", "unknown", "not sure", "nahi", "nahi hai", "no history"}
WEAK_EVIDENCE_STRENGTHS = {"location_with_context", "semantic_fallback"}
URINARY_SCOPE_CODES = {"E_55_@_V_185"}


def _info_prediction(reason: str, disease: str, message: str) -> List[Dict]:
    return [{
        "rank": 1,
        "disease": disease,
        "pathology": disease,
        "label_id": -1,
        "confidence": 0.0,
        "confidence_pct": "0.0%",
        "flag": "INFO",
        "source": "structured clinical evidence",
        "score_type": "not run",
        "warning": message,
        "reason": reason,
    }]


def _evidence_quality(inferred: List[Dict]) -> Dict:
    if not inferred:
        return {
            "status": "none",
            "message": "No official DDXPlus evidence was inferred from the text.",
        }

    strengths = {item.get("evidence_strength", "unknown") for item in inferred}
    if strengths and strengths <= WEAK_EVIDENCE_STRENGTHS:
        return {
            "status": "weak",
            "strengths": sorted(strengths),
            "message": (
                "Only weak evidence was inferred, such as a body location or "
                "semantic fallback, so the disease classifier was not run."
            ),
        }

    return {
        "status": "strong",
        "strengths": sorted(strengths),
        "message": "At least one direct symptom/evidence match was found.",
    }


def _scope_limitation(codes: List[str]) -> Dict | None:
    code_set = {str(code).upper() for code in codes or []}
    if code_set & URINARY_SCOPE_CODES:
        return {
            "status": "model_scope_limited",
            "reason": "urinary_evidence_outside_model_scope",
            "message": (
                "I understood urinary or urination-related evidence, but this "
                "classifier is trained only on the current 49-condition DDXPlus "
                "scope. I will not force a disease prediction from that model "
                "for this symptom pattern."
            ),
        }
    return None


def greeting_message() -> str:
    """Public accessor for the initial HealthBot greeting (used by the API layer)."""
    return _GREETING


def _is_reset(msg: str) -> bool:
    return msg.strip().lower() in _RESET_TRIGGERS


def _is_greeting_only(msg: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z\u0900-\u097F ]", " ", msg).strip().lower()
    return cleaned in _GREETING_WORDS


def _is_thanks_only(msg: str) -> bool:
    cleaned = re.sub(r"[^a-zA-Z\u0900-\u097F ]", " ", msg).strip().lower()
    return cleaned in _THANKS_WORDS


def _looks_like_question(msg: str) -> bool:
    text = (msg or "").strip().lower()
    if not text:
        return False
    if "?" in text:
        return True
    return any(text.startswith(prefix + " ") for prefix in _QUESTION_PREFIXES)


def _extract_duration(text: str) -> str | None:
    patterns = [
        r"\b\d+\s*(?:minute|minutes|min|hour|hours|hr|hrs|day|days|week|weeks|month|months)\b",
        r"\b\d+\s*(?:year|years)\b(?!\s*old\b)",
        r"\b(?:since|from)\s+(?:yesterday|today|this morning|morning|last night|tonight|last week|last month)\b",
        r"\b(?:few|couple of)\s+(?:minutes|hours|days|weeks|months|years)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def _extract_severity(text: str) -> str | None:
    lower = text.lower()
    if any(word in lower for word in ["mild", "हलका", "halka"]):
        return "mild"
    if any(word in lower for word in ["moderate", "medium", "मध्यम"]):
        return "moderate"
    if any(word in lower for word in ["severe", "तीव्र", "जास्त", "bahut", "khup"]):
        return "severe"

    scale_match = re.search(
        r"\b([1-9]|10)\s*(?:/10|out of 10)\b|\b(?:pain|severity|score)\s*(?:is|=|:)?\s*([1-9]|10)\b",
        lower,
    )
    if scale_match:
        score = int(scale_match.group(1) or scale_match.group(2))
        if score <= 3:
            return "mild"
        if score <= 6:
            return "moderate"
        return "severe"
    return None


def _age_group_from_age(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 18:
        return "child"
    if age >= 65:
        return "elderly"
    return "adult"


def _slot_label(slot: str) -> str:
    labels = {
        "main_symptoms": "main symptoms",
        "duration": "symptom duration",
        "severity": "symptom severity",
        "age_group": "age group",
        "age_years": "age",
        "sex": "gender",
        "pain_location": "pain location",
        "previous_disease": "previous disease/history",
        "family_history": "family history",
    }
    return labels.get(slot, slot.replace("_", " "))


def _display_sex(sex: str | None) -> str:
    if sex == "M":
        return "male"
    if sex == "F":
        return "female"
    return "unknown"


def _list_phrase(items: List[str]) -> str:
    clean = [str(item).strip() for item in items if str(item).strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    if len(clean) == 2:
        return f"{clean[0]} and {clean[1]}"
    return f"{', '.join(clean[:-1])}, and {clean[-1]}"


def _case_detail_summary(case_details: Dict) -> str:
    if not case_details:
        return ""

    gender = case_details.get("gender")
    if gender == "M":
        gender = "male"
    elif gender == "F":
        gender = "female"

    fields = [
        ("age", case_details.get("age")),
        ("gender", gender),
        ("symptoms", case_details.get("symptoms_text")),
        ("severity", case_details.get("severity")),
        ("pain location", case_details.get("pain_location")),
        ("duration", case_details.get("duration")),
        ("previous disease", case_details.get("previous_disease_or_history")),
        ("family history", case_details.get("genetic_or_family_history")),
    ]
    parts = [
        f"{label}: {value}"
        for label, value in fields
        if value is not None and str(value).strip() and str(value).strip().lower() != "unknown"
    ]
    return "; ".join(parts)


def _is_slot_only_answer(message: str, normalization: Dict) -> bool:
    if normalization.get("mapped_symptoms"):
        return False
    lower = message.lower()
    age, sex = parse_age_sex_from_text(message)
    has_slot_info = bool(_extract_duration(lower) or _extract_severity(lower) or age is not None or sex != "U")
    token_count = len(re.findall(r"[\u0900-\u097Fa-zA-Z0-9]+", message))
    return has_slot_info and token_count <= 6


def confidence_status(predictions: List[Dict]) -> str:
    if not predictions:
        return "unknown"
    confidence = float(predictions[0].get("confidence", 0.0))
    if confidence >= 0.60:
        return "high"
    if confidence >= CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def _merge_inferred_evidences(primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    seen = set()
    for item in list(primary or []) + list(secondary or []):
        key = (item.get("code"), item.get("negated"), item.get("meaning"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def build_assessment_payload(
    original_text: str,
    predictions: List[Dict],
    normalization: Dict,
    red_flag_result: Dict,
    missing_info: List[str] | None = None,
    evidence_bridge: Dict | None = None,
) -> Dict:
    mapped = normalization.get("mapped_symptoms", [])
    normalized_symptoms = [item.get("canonical", "") for item in mapped if item.get("canonical")]
    if not normalized_symptoms:
        normalized_symptoms = [normalization.get("normalized_text", original_text)]

    return {
        "original_text": original_text,
        "detected_language": normalization.get("detected_language", {}),
        "normalized_symptoms": list(dict.fromkeys(normalized_symptoms)),
        "normalized_text": normalization.get("normalized_text", original_text),
        "mapped_symptoms": mapped,
        "predictions": predictions,
        "red_flag_result": red_flag_result,
        "evidence_bridge": evidence_bridge or {},
        "confidence_status": confidence_status(predictions),
        "missing_info": missing_info or [],
        "disclaimer": (
            "This tool is for educational information only and does not replace "
            "professional medical advice, diagnosis, or treatment."
        ),
    }


def format_assessment(payload: Dict) -> Tuple[str, str]:
    """Return (formatted_response, formatter_used)."""
    ollama_response = format_with_ollama(payload)
    if ollama_response:
        return ollama_response, "Ollama"
    return summarize_response(payload), "Template"


def assess_symptoms(
    symptoms_text: str,
    top_n: int = 5,
    missing_info: List[str] | None = None,
    case_details: Dict | None = None,
) -> Dict:
    """Run the complete local non-RAG assessment pipeline."""
    normalization = normalize_symptoms(symptoms_text)
    red_flag_result = detect_red_flags(symptoms_text)
    model_input = normalization.get("normalized_text") or symptoms_text
    inferred_all = infer_evidence_codes_from_text(symptoms_text, include_denied=True)
    normalized_for_evidence = str(normalization.get("normalized_text") or "").strip()
    has_denied_evidence = any(item.get("negated") for item in inferred_all)
    if (
        normalized_for_evidence
        and normalized_for_evidence.lower() != str(symptoms_text or "").strip().lower()
        and not has_denied_evidence
    ):
        normalized_inferred = infer_evidence_codes_from_text(normalized_for_evidence, include_denied=True)
        inferred_all = _merge_inferred_evidences(inferred_all, normalized_inferred)
    inferred = [item for item in inferred_all if not item.get("negated")]
    denied = [item for item in inferred_all if item.get("negated")]
    evidence_bridge = {}

    if inferred:
        codes = [item["code"] for item in inferred]
        age, sex = parse_age_sex_from_text(symptoms_text)
        initial = select_initial_evidence(inferred)
        quality = _evidence_quality(inferred)
        scope = _scope_limitation(codes)

        if quality["status"] == "weak":
            predictions = _info_prediction(
                reason="weak_evidence",
                disease="More specific symptom detail needed",
                message=quality["message"],
            )
            search_results = []
            mode = "weak_evidence"
        elif scope:
            predictions = _info_prediction(
                reason=scope["reason"],
                disease="Model scope limitation",
                message=scope["message"],
            )
            search_results = []
            mode = "model_scope_limited"
        else:
            # The structured DDXPlus engine is the single canonical disease
            # predictor. Conversation slots remain in the assessment context
            # for safety, triage, response wording, and future questioning;
            # their presence must not switch prediction models.
            predictions = predict_case(
                age=age,
                sex=sex,
                evidences=codes,
                initial_evidence=initial,
                top_n=top_n,
            )
            search_results = semantic_search_case(
                age=age,
                sex=sex,
                evidences=codes,
                initial_evidence=initial,
                top_k=3,
            )
            mode = "human_symptoms_to_structured_evidence"

        evidence_bridge = {
            "mode": mode,
            "age": age,
            "sex": sex,
            "initial_evidence": initial,
            "inferred_evidences": inferred,
            "denied_evidences": denied,
            "evidence_quality": quality,
            "scope_warning": scope,
            "case_details": case_details or {},
            "classifier_note": "",
            "biobert": biobert_status(),
        }
    else:
        # No usable evidence at all (denied-only mentions don't count as
        # usable evidence either). `predict()`/`semantic_search()` already
        # short-circuit to an explicit "evidence required" result for plain
        # text with no recognizable evidence codes — see model/predictor.py.
        predictions = predict(model_input, top_n=top_n)
        search_results = semantic_search(model_input, top_k=3)
        evidence_bridge = {
            "mode": "free_text_fallback",
            "stall_reason": "no_official_ddxplus_evidence_inferred",
            "message": (
                "Natural-language text did not map to official DDXPlus evidence. "
                "The structured classifier was not run on a clinical evidence case."
            ),
            "denied_evidences": denied,
            "biobert": biobert_status(),
        }

    severity_result = evaluate_differential_severity(predictions)
    evidence_bridge["severity_triage"] = {
        "any_high_acuity_candidate": severity_result.any_high_acuity_candidate,
        "weighted_severity_score": severity_result.weighted_severity_score,
        "message": severity_result.message,
    }

    payload = build_assessment_payload(
        original_text=symptoms_text,
        predictions=predictions,
        normalization=normalization,
        red_flag_result=red_flag_result,
        missing_info=missing_info,
        evidence_bridge=evidence_bridge,
    )
    formatted_response, formatter_used = format_assessment(payload)

    return {
        "normalization": normalization,
        "red_flag_result": red_flag_result,
        "predictions": predictions,
        "semantic_search": search_results,
        "evidence_bridge": evidence_bridge,
        "payload": payload,
        "formatted_response": formatted_response,
        "formatter_used": formatter_used,
        "confidence_status": payload["confidence_status"],
    }


def assessment_explanation(assessment: Dict | None, top_n_features: int = 6) -> Dict:
    """
    Bridge an `assess_symptoms()` result to `model.predictor.explain_case()`
    for the top-ranked prediction.

    Only available when the evidence bridge actually resolved structured
    DDXPlus evidence codes from the user's text (mode
    "human_symptoms_to_structured_evidence") — free-text-fallback cases have
    no evidence codes to explain against, and an empty/missing assessment
    has nothing to explain at all. In both cases this returns a `note`
    instead of contributions, rather than raising or silently guessing.
    """
    if not assessment or not assessment.get("predictions"):
        return {"pathology": None, "contributions": [], "note": "No assessment available yet."}

    evidence_bridge = assessment.get("evidence_bridge") or {}
    if evidence_bridge.get("mode") != "human_symptoms_to_structured_evidence":
        return {
            "pathology": assessment["predictions"][0].get("disease"),
            "contributions": [],
            "note": (
                "The symptom text did not map to official DDXPlus evidence codes, "
                "so there is no structured evidence to attribute this prediction to."
            ),
        }

    codes = [item["code"] for item in evidence_bridge.get("inferred_evidences", [])]
    return explain_case(
        age=evidence_bridge.get("age"),
        sex=evidence_bridge.get("sex"),
        evidences=codes,
        initial_evidence=evidence_bridge.get("initial_evidence"),
        pathology=assessment["predictions"][0].get("pathology"),
        top_k_features=top_n_features,
    )


class ChatSession:
    """Conversation state for one HealthBot chat session."""

    def __init__(self):
        self.state = STATE_GREETING
        self.symptom_messages: List[str] = []
        self.clarification_turns = 0
        self.pending_slot: str | None = None
        self.last_assessment: Dict | None = None
        self.slots = {
            "main_symptoms": [],
            "duration": None,
            "severity": None,
            "age_group": None,
            "age_years": None,
            "sex": None,
            "fever": None,
            "pain_location": None,
            "breathing_problem": None,
            "chronic_conditions": None,
            "previous_disease": None,
            "family_history": None,
            "medication_allergy": None,
            "red_flags": [],
            "language": None,
        }

    def reset(self):
        self.__init__()

    def _combined_text(self, message: str) -> str:
        return " ".join(self.symptom_messages + [message]).strip()

    def _context_summary(self) -> str:
        parts = []
        main_symptoms = self.slots.get("main_symptoms") or []
        if main_symptoms:
            parts.append(f"main symptoms: {_list_phrase(main_symptoms)}")
        if self.slots.get("duration"):
            parts.append(f"duration: {self.slots['duration']}")
        if self.slots.get("severity"):
            parts.append(f"severity: {self.slots['severity']}")
        if self.slots.get("age_years") is not None:
            parts.append(f"age: {self.slots['age_years']}")
        elif self.slots.get("age_group"):
            parts.append(f"age group: {self.slots['age_group']}")
        if self.slots.get("sex"):
            parts.append(f"gender: {_display_sex(self.slots['sex'])}")
        if self.slots.get("pain_location"):
            parts.append(f"pain location: {self.slots['pain_location']}")
        if self.slots.get("previous_disease"):
            parts.append(f"previous disease/history: {self.slots['previous_disease']}")
        if self.slots.get("family_history"):
            parts.append(f"family history: {self.slots['family_history']}")
        return "; ".join(parts)

    def _assembled_case_text(self) -> str:
        segments: List[str] = []
        seen = set()
        for item in self.symptom_messages:
            text = str(item).strip()
            if text and text not in seen:
                segments.append(text)
                seen.add(text)

        main_symptoms = self.slots.get("main_symptoms") or []
        if main_symptoms:
            phrase = _list_phrase(main_symptoms)
            if phrase and phrase not in seen:
                segments.append(phrase)
                seen.add(phrase)

        if self.slots.get("duration"):
            segments.append(f"duration {self.slots['duration']}")
        if self.slots.get("severity"):
            segments.append(f"severity {self.slots['severity']}")
        if self.slots.get("age_years") is not None:
            segments.append(f"age {self.slots['age_years']}")
        elif self.slots.get("age_group"):
            segments.append(self.slots["age_group"])
        if self.slots.get("sex"):
            segments.append(f"gender {_display_sex(self.slots['sex'])}")
        if self.slots.get("pain_location"):
            segments.append(f"pain location {self.slots['pain_location']}")
        if self.slots.get("previous_disease"):
            segments.append(f"previous disease {self.slots['previous_disease']}")
        if self.slots.get("family_history"):
            segments.append(f"family history {self.slots['family_history']}")
        return ". ".join(segment for segment in segments if segment).strip()

    def _severity_focus(self) -> str:
        main_symptoms = self.slots.get("main_symptoms") or []
        for symptom in main_symptoms:
            if "urination" in symptom or "urine" in symptom:
                return symptom
        if self.slots.get("pain_location"):
            return f"{self.slots['pain_location']} pain"
        if self.slots.get("breathing_problem"):
            return "breathing difficulty"
        if self.slots.get("fever"):
            return "fever"
        if main_symptoms:
            return main_symptoms[0]
        return "symptoms"

    def _should_collect_context(self) -> bool:
        return bool(self._missing_required_slots())

    def _pending_slot_prompt(self, slot: str) -> str:
        if slot == "main_symptoms":
            context = self._context_summary()
            if context:
                return f"I have noted {context}. What is the main symptom complaint you want me to focus on?"
            return "Please describe the main symptoms in one sentence."
        if slot == "age_years":
            return "What is the patient's age?"
        if slot == "sex":
            return "What is the patient's gender?"
        if slot == "pain_location":
            symptoms = _list_phrase(self.slots.get("main_symptoms") or [])
            if symptoms:
                return (
                    f"I noted **{symptoms}**. Where is the main pain or discomfort located? "
                    "Reply `none` if there is no pain, or use the actual area like `head` or `throat`. "
                    "Only say `chest` if there is chest pain or chest discomfort too."
                )
            return (
                "Where is the main pain or discomfort located? Reply `none` if there is no pain. "
                "Only say `chest` if there is chest pain or chest discomfort."
            )
        if slot == "previous_disease":
            return "Do you have any previous disease or medical history, such as diabetes, asthma, high BP, heart disease, or none?"
        if slot == "family_history":
            return "Do you have any family history such as heart disease, asthma, allergy, or none?"
        if slot == "duration":
            focus = self._severity_focus()
            return f"How long have you had the {focus}?"
        if slot == "severity":
            focus = self._severity_focus()
            return f"How severe is the {focus} right now?"
        if slot == "age_group":
            if self.slots.get("age_years") is not None:
                return f"I captured age {self.slots['age_years']}. Is this for a child, adult, or elderly person?"
            return "Is this for a child, adult, or elderly person?"
        return "Could you share one more detail about your symptoms?"

    def _update_slots(self, message: str, normalization: Dict, red_flags: Dict) -> None:
        lower_msg = message.lower()
        normalized_text = normalization.get("normalized_text", "").lower()
        detected_language = normalization.get("detected_language", {})
        llm_slots = (normalization.get("llm_extraction") or {}).get("slots") or {}

        self.slots["language"] = detected_language.get("language", "unknown")
        pending = self.pending_slot

        bare_number = re.fullmatch(r"\s*(\d{1,3})\s*", message or "")
        if pending == "age_years" and bare_number:
            age_value = int(bare_number.group(1))
            if 0 <= age_value <= 110:
                self.slots["age_years"] = age_value
                self.slots["age_group"] = _age_group_from_age(age_value)

        if pending == "duration" and bare_number:
            self.slots["duration"] = f"{bare_number.group(1)} days"

        if pending == "sex":
            cleaned = lower_msg.strip()
            if cleaned in {"m", "male", "man", "boy"}:
                self.slots["sex"] = "M"
            elif cleaned in {"f", "female", "woman", "girl"}:
                self.slots["sex"] = "F"

        if pending == "pain_location":
            cleaned = lower_msg.strip()
            self.slots["pain_location"] = "none" if cleaned in NONE_VALUES else cleaned

        if pending == "previous_disease":
            cleaned = lower_msg.strip()
            value = "none" if cleaned in NONE_VALUES else message.strip()
            self.slots["previous_disease"] = value
            self.slots["chronic_conditions"] = value

        if pending == "family_history":
            cleaned = lower_msg.strip()
            self.slots["family_history"] = "none" if cleaned in NONE_VALUES else message.strip()

        if llm_slots:
            age_value = llm_slots.get("age")
            if self.slots.get("age_years") is None and isinstance(age_value, int):
                self.slots["age_years"] = age_value
                self.slots["age_group"] = _age_group_from_age(age_value)
            if not self.slots.get("sex") and llm_slots.get("gender") in {"M", "F"}:
                self.slots["sex"] = llm_slots["gender"]
            if not self.slots.get("duration") and llm_slots.get("duration"):
                self.slots["duration"] = llm_slots["duration"]
            if not self.slots.get("severity") and llm_slots.get("severity"):
                self.slots["severity"] = llm_slots["severity"]
            if not self.slots.get("pain_location") and llm_slots.get("pain_location"):
                self.slots["pain_location"] = llm_slots["pain_location"]
            if not self.slots.get("previous_disease") and llm_slots.get("previous_disease"):
                self.slots["previous_disease"] = llm_slots["previous_disease"]
                self.slots["chronic_conditions"] = llm_slots["previous_disease"]
            if not self.slots.get("family_history") and llm_slots.get("family_history"):
                self.slots["family_history"] = llm_slots["family_history"]

        mapped = normalization.get("mapped_symptoms", [])
        symptoms = [item["canonical"] for item in mapped if item.get("canonical")]
        if symptoms:
            existing = self.slots["main_symptoms"] or []
            self.slots["main_symptoms"] = list(dict.fromkeys(existing + symptoms))
        elif (
            normalized_text
            and not _looks_like_question(message)
            and len(re.findall(r"[\u0900-\u097Fa-zA-Z]+", message)) >= 3
            and not _is_slot_only_answer(message, normalization)
        ):
            self.slots["main_symptoms"] = self.slots["main_symptoms"] or [normalized_text]

        duration = _extract_duration(lower_msg)
        if duration:
            self.slots["duration"] = duration

        severity = _extract_severity(lower_msg)
        if severity:
            self.slots["severity"] = severity

        age, sex = parse_age_sex_from_text(message)
        if age is not None:
            self.slots["age_years"] = age
            self.slots["age_group"] = _age_group_from_age(age)
        if sex != "U":
            self.slots["sex"] = sex

        if any(word in lower_msg for word in ["child", "baby", "infant", "kid", "लहान", "बाळ"]):
            self.slots["age_group"] = "child"
        elif any(word in lower_msg for word in ["elderly", "senior", "old age", "वृद्ध"]):
            self.slots["age_group"] = "elderly"
        elif any(word in lower_msg for word in ["adult", "वयस्क"]):
            self.slots["age_group"] = "adult"

        self.slots["fever"] = self.slots["fever"] or ("fever" in normalized_text)
        self.slots["breathing_problem"] = self.slots["breathing_problem"] or any(
            phrase in normalized_text for phrase in ["difficulty breathing", "shortness of breath"]
        )

        if any(word in lower_msg for word in ["diabetes", "asthma", "hypertension", "bp", "thyroid"]):
            self.slots["chronic_conditions"] = "mentioned"
            if not self.slots.get("previous_disease"):
                self.slots["previous_disease"] = message.strip()

        if "allergy" in lower_msg or "allergic" in lower_msg:
            self.slots["medication_allergy"] = "mentioned"

        self.slots["red_flags"] = red_flags.get("matched_flags", [])

    def _slot_is_filled(self, slot: str) -> bool:
        value = self.slots.get(slot)
        if isinstance(value, list):
            return bool(value)
        return value is not None and value != ""

    def _missing_slots(self, require_age_group: bool = False) -> List[str]:
        missing = []
        if not self.slots["main_symptoms"]:
            missing.append("main_symptoms")
        if not self.slots["duration"]:
            missing.append("duration")
        if not self.slots["severity"] and (
            self.slots["pain_location"] or self.slots["breathing_problem"] or self.slots["fever"]
        ):
            missing.append("severity")
        if require_age_group and not self.slots["age_group"]:
            missing.append("age_group")
        return missing

    def _missing_required_slots(self) -> List[str]:
        if not self._slot_is_filled("main_symptoms"):
            return ["main_symptoms"]
        return [slot for slot in REQUIRED_DETAIL_SLOTS if not self._slot_is_filled(slot)]

    def _case_details(self) -> Dict:
        symptoms = _list_phrase(self.slots.get("main_symptoms") or [])
        return {
            "age": self.slots.get("age_years") or "unknown",
            "gender": self.slots.get("sex") or "unknown",
            "symptoms_text": symptoms or self._assembled_case_text(),
            "duration": self.slots.get("duration") or "unknown",
            "severity": self.slots.get("severity") or "unknown",
            "pain_location": self.slots.get("pain_location") or "unknown",
            "previous_disease_or_history": self.slots.get("previous_disease") or "unknown",
            "genetic_or_family_history": self.slots.get("family_history") or "unknown",
        }

    def _question_for_slot(self, slot: str) -> str:
        return self._pending_slot_prompt(slot)

    def _slot_hint(self, slot: str) -> str:
        return _SLOT_HINTS.get(slot, "Please share one more detail.")

    def _message_should_extend_case_text(self, message: str, normalization: Dict) -> bool:
        if normalization.get("mapped_symptoms"):
            return True
        if _looks_like_question(message) or _is_greeting_only(message) or _is_thanks_only(message):
            return False
        if _is_slot_only_answer(message, normalization):
            return False
        tokens = re.findall(r"[\u0900-\u097Fa-zA-Z]+", message)
        return len(tokens) >= 3

    def _next_detail_to_ask(self, low_confidence: bool) -> str | None:
        missing = self._missing_required_slots()
        return missing[0] if missing else None

    def _reply_for_question(self, user_msg: str) -> str | None:
        lower = user_msg.lower().strip()

        if any(phrase in lower for phrase in ["what can you do", "how to use", "help", "how should i ask"]):
            return (
                "I can take your symptoms in natural language, look for warning signs, "
                "and show the closest model matches from your local clinical dataset.\n\n"
                f"{_HELP_HINT}"
            )

        if any(phrase in lower for phrase in ["what detail", "what more", "what info", "what do you need"]):
            slot = self.pending_slot or self._next_detail_to_ask(low_confidence=True) or "main_symptoms"
            return (
                f"The most useful next detail is the **{_slot_label(slot)}**.\n\n"
                f"{self._question_for_slot(slot)}\n\n{self._slot_hint(slot)}"
            )

        if "confidence" in lower or "why low" in lower:
            return (
                "Low confidence usually means the current symptom pattern overlaps several conditions "
                "or there is not enough detail yet for the model to separate them clearly. "
                "Duration, severity, age, and a clearer symptom description usually help."
            )

        if any(phrase in lower for phrase in ["what do you think", "what could it be", "possible condition", "what is it"]):
            if self.last_assessment:
                missing = self._missing_slots(require_age_group=False)
                return self._chat_assessment_reply(self.last_assessment, missing, include_reset_hint=False)
            if self.symptom_messages:
                combined_text = " ".join(self.symptom_messages).strip()
                assessment = assess_symptoms(combined_text, top_n=5, missing_info=self._missing_slots())
                self.last_assessment = assessment
                return self._chat_assessment_reply(assessment, self._missing_slots(), include_reset_hint=False)
            return f"I can help with that once you share the symptoms first.\n\n{_HELP_HINT}"

        return None

    def _chat_assessment_reply(
        self,
        assessment: Dict,
        missing: List[str],
        include_reset_hint: bool = True,
    ) -> str:
        predictions = assessment.get("predictions", [])
        top = predictions[0] if predictions else {}
        red_flag = assessment.get("red_flag_result", {}) or {}
        confidence = assessment.get("confidence_status", "unknown")
        evidence_bridge = assessment.get("evidence_bridge") or {}
        case_details = evidence_bridge.get("case_details") or {}
        case_detail_summary = _case_detail_summary(case_details)
        lines: List[str] = []

        if red_flag.get("has_red_flag"):
            lines.append(red_flag.get("safety_message", "A warning sign was detected."))

        severity_triage = evidence_bridge.get("severity_triage") or {}
        if severity_triage.get("any_high_acuity_candidate") and not red_flag.get("has_red_flag"):
            if confidence != "high":
                lines.append(
                    "A high-acuity condition appears in the broader model candidate list, "
                    "but the current model confidence is not high, so I am not treating it as a reliable prediction. "
                    "Seek urgent care if symptoms feel severe, are worsening, or include chest pain or breathing trouble."
                )
            else:
                # Independent of keyword red flags: this uses the official
                # DDXPlus per-pathology severity rating for the candidate list.
                lines.append(severity_triage.get("message", ""))

        if case_detail_summary and top.get("flag") not in {"INFO", "ERROR"}:
            lines.append(f"Details used by model: {case_detail_summary}.")

        if predictions:
            if top.get("flag") in {"INFO", "ERROR"}:
                lines.append(top.get("warning", "The model was not run for this message."))
            else:
                top3 = ", ".join(
                    f"{item.get('disease', 'Unknown')} ({item.get('confidence_pct', 'n/a')})"
                    for item in predictions[:3]
                )
                other_matches = ", ".join(
                    f"{item.get('disease', 'Unknown')} ({item.get('confidence_pct', 'n/a')})"
                    for item in predictions[1:3]
                )
                if confidence == "high":
                    lines.append(
                        f"Based on what you described, the strongest model match is **{top.get('disease', 'Unknown')}** "
                        f"with {top.get('confidence_pct', 'n/a')} confidence."
                    )
                elif confidence == "medium":
                    lines.append(
                        f"Based on your symptoms, the closest model matches are {top3}. "
                        f"The top match is **{top.get('disease', 'Unknown')}**."
                    )
                else:
                    other_text = f" Other close matches are {other_matches}." if other_matches else ""
                    low_confidence_message = (
                        f"The top model match is **{top.get('disease', 'Unknown')}** "
                        f"({top.get('confidence_pct', 'n/a')}), but confidence is low. "
                        f"{other_text} Treat these as weak matches rather than a reliable answer."
                    )
                    lines.append(low_confidence_message.replace("  ", " "))
        else:
            lines.append("I could not form a reliable model match from the current message.")

        if confidence == "low" and missing:
            next_slot = self._next_detail_to_ask(low_confidence=True)
            if next_slot:
                lines.append(
                    f"To refine it, the most useful next detail is the **{_slot_label(next_slot)}**. "
                    f"{self._slot_hint(next_slot)}"
                )
        elif not red_flag.get("has_red_flag"):
            lines.append(
                "No emergency red flag was detected from the current text, but this is still not a medical diagnosis."
            )

        if include_reset_hint:
            lines.append("Type `reset` if you want to start a new case.")

        return "\n\n".join(lines)

    def _slot_update_reply(self, changed_slots: List[str]) -> str:
        labels = ", ".join(_slot_label(slot) for slot in changed_slots)
        if self.last_assessment and self.last_assessment.get("predictions"):
            top = self.last_assessment["predictions"][0]
            next_slot = self._next_detail_to_ask(low_confidence=True)
            if next_slot and next_slot not in changed_slots and self._should_collect_context():
                self.pending_slot = next_slot
                return (
                    f"Thanks, I noted the **{labels}**.\n\n"
                    f"The current top model match is **{top.get('disease', 'Unknown')}** "
                    f"({top.get('confidence_pct', 'n/a')}). The next most useful detail is the **{_slot_label(next_slot)}**.\n\n"
                    f"{self._question_for_slot(next_slot)}\n\n{self._slot_hint(next_slot)}"
                )
            return (
                f"Thanks, I noted the **{labels}**.\n\n"
                f"The current top model match is still **{top.get('disease', 'Unknown')}** "
                f"({top.get('confidence_pct', 'n/a')}). "
                "You can add another symptom or ask `what do you think?` for a quick summary."
            )
        next_slot = self._next_detail_to_ask(low_confidence=True)
        if next_slot and next_slot not in changed_slots:
            self.pending_slot = next_slot
            return (
                f"Thanks, I noted the **{labels}**.\n\n"
                f"The next most useful detail is the **{_slot_label(next_slot)}**.\n\n"
                f"{self._question_for_slot(next_slot)}\n\n{self._slot_hint(next_slot)}"
            )
        return (
            f"Thanks, I noted the **{labels}**.\n\n"
            "You can add another symptom or ask `what do you think?` and I will summarize the current case."
        )

    def reply(self, user_msg: str) -> str:
        user_msg = (user_msg or "").strip()

        if not user_msg:
            return "Please type your symptoms so I can help."

        if _is_reset(user_msg):
            self.reset()
            return _GREETING

        if self.state == STATE_GREETING:
            self.state = STATE_COLLECTING
            if _is_greeting_only(user_msg):
                return _GREETING

        if _is_greeting_only(user_msg) and not self.symptom_messages:
            return f"{_GREETING}\n\n{_HELP_HINT}"

        if _is_thanks_only(user_msg):
            return "You're welcome. Share more symptoms anytime and I will continue from here."

        question_reply = None
        msg_normalization = normalize_symptoms(user_msg)
        extends_case_text = self._message_should_extend_case_text(user_msg, msg_normalization)
        if _looks_like_question(user_msg) and not extends_case_text:
            question_reply = self._reply_for_question(user_msg)
        if question_reply:
            return question_reply

        msg_red_flags = detect_red_flags(user_msg)
        previous_slots = {
            "duration": self.slots["duration"],
            "severity": self.slots["severity"],
            "age_group": self.slots["age_group"],
            "age_years": self.slots["age_years"],
            "sex": self.slots["sex"],
            "pain_location": self.slots["pain_location"],
            "previous_disease": self.slots["previous_disease"],
            "family_history": self.slots["family_history"],
        }
        self._update_slots(user_msg, msg_normalization, msg_red_flags)
        changed_slots = [
            slot
            for slot, previous_value in previous_slots.items()
            if previous_value != self.slots[slot] and self._slot_is_filled(slot)
        ]

        if extends_case_text and user_msg not in self.symptom_messages:
            self.symptom_messages.append(user_msg)

        if self.pending_slot and not self._slot_is_filled(self.pending_slot):
            if self.clarification_turns >= MAX_CLARIFICATION_TURNS:
                unresolved_slot = self.pending_slot
                combined_text = self._assembled_case_text()
                if not combined_text:
                    return self._question_for_slot(unresolved_slot)
                assessment = assess_symptoms(combined_text, top_n=5, missing_info=self._missing_slots())
                self.last_assessment = assessment
                self.pending_slot = None
                return (
                    f"I still could not clearly capture the **{_slot_label(unresolved_slot)}**, "
                    "so I am giving the best answer from the details I already have.\n\n"
                    + self._chat_assessment_reply(assessment, self._missing_slots(), include_reset_hint=False)
                )
            self.clarification_turns += 1
            return (
                f"I could not understand the **{_slot_label(self.pending_slot)}** from that message.\n\n"
                f"{self._slot_hint(self.pending_slot)}"
            )
        self.pending_slot = None

        if not extends_case_text and changed_slots and self.symptom_messages and self._missing_required_slots():
            return self._slot_update_reply(changed_slots)

        if not extends_case_text and not changed_slots and not _looks_like_question(user_msg):
            if self.symptom_messages:
                return (
                    "I did not catch a clear new symptom or detail from that message.\n\n"
                    "You can tell me another symptom, or give details like `2 days`, `moderate`, or `45 years old`."
                )
            return f"I need the symptom details first.\n\n{_HELP_HINT}"

        combined_text = " ".join(self.symptom_messages).strip()
        combined_text = self._assembled_case_text() or combined_text
        if not combined_text:
            return f"I need the symptom details first.\n\n{_HELP_HINT}"

        combined_normalization = normalize_symptoms(combined_text)
        combined_red_flags = detect_red_flags(combined_text)

        normalized_text = combined_normalization.get("normalized_text", "")
        has_enough_symptoms = bool(self.slots["main_symptoms"]) and len(normalized_text.split()) >= 1
        if not has_enough_symptoms and self.clarification_turns < MAX_CLARIFICATION_TURNS:
            self.clarification_turns += 1
            self.pending_slot = "main_symptoms"
            return self._question_for_slot("main_symptoms")

        next_slot = self._next_detail_to_ask(low_confidence=True)
        if next_slot:
            self.pending_slot = next_slot
            intro = "I need a few clinical details before running the disease classifier."
            if combined_red_flags.get("has_red_flag"):
                intro = (
                    combined_red_flags.get("safety_message", "A warning sign was detected.")
                    + "\n\nI will still collect the key details for the model, but urgent symptoms should be handled promptly."
                )
            return f"{intro}\n\n**{self._question_for_slot(next_slot)}**\n\n{self._slot_hint(next_slot)}"

        missing = []
        case_details = self._case_details()

        if RED_FLAG_FIRST and combined_red_flags.get("has_red_flag"):
            assessment = assess_symptoms(combined_text, top_n=5, missing_info=missing, case_details=case_details)
            self.last_assessment = assessment
            self.state = STATE_DONE
            return self._chat_assessment_reply(assessment, missing)

        assessment = assess_symptoms(combined_text, top_n=5, missing_info=missing, case_details=case_details)
        self.last_assessment = assessment
        self.state = STATE_DONE
        return self._chat_assessment_reply(assessment, missing)
