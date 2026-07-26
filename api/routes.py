"""
api/routes.py
──────────────
Flask Blueprint exposing the existing NLP/ML pipeline (chatbot, structured
predictor, severity engine, explainability) as a small, REST-ish JSON API.

Design notes
────────────
- This layer is intentionally thin. It does not contain any clinical or
  ML logic — it only translates HTTP requests into calls against
  `chatbot.bot`, `model.predictor`, and `chatbot.session_store`, and
  serializes their results back to JSON. All actual reasoning still lives
  in the modules under `utils/`, `model/`, and `chatbot/bot.py`.
- The chat endpoints are session-based (`session_id`) because
  `ChatSession` is a multi-turn, stateful object — see
  `chatbot/session_store.py` for how that state is kept across stateless
  HTTP requests.
- `/api/analyze` and `/api/case/predict` are stateless, single-shot
  endpoints: useful for the "quick analyze" examples in the UI and for
  exercising the structured ML model directly without going through the
  conversational slot-filling flow.
"""

from __future__ import annotations

from typing import Any, Dict, List

from flask import Blueprint, jsonify, request

from chatbot.bot import assess_symptoms, assessment_explanation, greeting_message
from chatbot.session_store import session_store
from config import APP_TITLE, EXPLAIN_TOP_K_FEATURES
from model.predict_disease import disease_classifier_metadata
from model.predictor import explain_case, known_pathology_names, model_metadata, predict_case, semantic_search_case
from utils.biobert_embedder import biobert_status
from utils.ddxplus_decoder import (
    evidence_markdown,
    infer_evidence_codes_from_text,
    parse_age_sex_from_text,
    resolve_evidence_input,
    resolve_initial_evidence_input,
    select_initial_evidence,
)
from utils.ollama_nlu_extractor import ollama_nlu_status

api = Blueprint("api", __name__, url_prefix="/api")


# ── Helpers ──────────────────────────────────────────────────────────────


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _serialize_assessment(assessment: Dict[str, Any], include_explanation: bool = True) -> Dict[str, Any]:
    """Flatten an `assess_symptoms()` result into a compact, frontend-friendly shape."""
    normalization = assessment.get("normalization", {})
    red_flag = assessment.get("red_flag_result", {})
    evidence_bridge = assessment.get("evidence_bridge", {})
    severity = evidence_bridge.get("severity_triage", {})

    meta: Dict[str, Any] = {
        "confidence_status": assessment.get("confidence_status"),
        "predictions": assessment.get("predictions", [])[:5],
        "language": normalization.get("detected_language", {}),
        "normalized_symptoms": [
            item.get("canonical") for item in normalization.get("mapped_symptoms", []) if item.get("canonical")
        ],
        "evidence_understood": [
            item.get("meaning") for item in evidence_bridge.get("inferred_evidences", []) if item.get("meaning")
        ],
        "denied_evidence": [
            item.get("meaning") for item in evidence_bridge.get("denied_evidences", []) if item.get("meaning")
        ],
        "red_flag": {
            "has_red_flag": red_flag.get("has_red_flag", False),
            "urgency_level": red_flag.get("urgency_level", "routine"),
            "matched_flags": red_flag.get("matched_flags", []),
            "safety_message": red_flag.get("safety_message", ""),
        },
        "severity_triage": {
            "any_high_acuity_candidate": severity.get("any_high_acuity_candidate", False),
            "message": severity.get("message", ""),
        },
        "evidence_mode": evidence_bridge.get("mode"),
        "evidence_quality": evidence_bridge.get("evidence_quality", {}),
        "scope_warning": evidence_bridge.get("scope_warning"),
    }

    if include_explanation:
        meta["explanation"] = assessment_explanation(assessment, top_n_features=EXPLAIN_TOP_K_FEATURES)

    return meta


# ── Health & metadata ────────────────────────────────────────────────────


@api.get("/health")
def health():
    return jsonify({"status": "ok", "app": APP_TITLE})


@api.get("/meta")
def meta():
    metadata = model_metadata()
    return jsonify({
        "app_title": APP_TITLE,
        "selected_model": metadata.get("selected_model"),
        "trained_at": metadata.get("trained_at"),
        "train_rows_used": metadata.get("train_rows_used"),
        "pathology_count": len(metadata.get("classes", [])),
        "pathologies": sorted(known_pathology_names()),
        "input_features": metadata.get("input_features", []),
        "simplified_classifier": disease_classifier_metadata(),
        "biobert": biobert_status(),
        "ollama_nlu": ollama_nlu_status(),
        "active_sessions": session_store.active_count(),
    })


# ── Conversational chatbot endpoints ─────────────────────────────────────


@api.post("/chat/start")
def chat_start():
    session_id = session_store.create()
    return jsonify({"session_id": session_id, "reply": greeting_message()})


@api.post("/chat/message")
def chat_message():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return _error("`message` is required.")

    session_id, session = session_store.get(body.get("session_id"))
    reply_text = session.reply(message)

    response: Dict[str, Any] = {"session_id": session_id, "reply": reply_text}
    if session.last_assessment is not None:
        response["meta"] = _serialize_assessment(session.last_assessment)
    return jsonify(response)


@api.post("/chat/reset")
def chat_reset():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id")
    if not session_id:
        return _error("`session_id` is required.")
    session_store.reset(session_id)
    return jsonify({"session_id": session_id, "reply": greeting_message()})


# ── Stateless one-shot analysis (used by example chips / quick analyze) ──


@api.post("/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return _error("`text` is required.")

    top_n = int(body.get("top_n", 5))
    assessment = assess_symptoms(text, top_n=top_n)
    return jsonify({
        "formatted_response": assessment["formatted_response"],
        "formatter_used": assessment["formatter_used"],
        **_serialize_assessment(assessment),
    })


# ── Structured clinical-case endpoints (advanced / developer panel) ──────


def _coerce_evidences(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    codes, _ = resolve_evidence_input(str(raw or ""))
    return codes


@api.post("/case/predict")
def case_predict():
    body = request.get_json(silent=True) or {}
    age = body.get("age")
    sex = body.get("sex", "")
    initial_evidence_raw = str(body.get("initial_evidence", ""))
    codes = _coerce_evidences(body.get("evidences"))
    top_n = int(body.get("top_n", 5))

    if not codes:
        return _error(
            "No recognizable symptoms in `evidences`. Try words like "
            "'cough, fever, chest pain, sore throat, nausea'."
        )

    initial_code, _ = resolve_initial_evidence_input(initial_evidence_raw, codes)
    predictions = predict_case(age=age, sex=sex, evidences=codes, initial_evidence=initial_code, top_n=top_n)
    explanation = explain_case(
        age=age,
        sex=sex,
        evidences=codes,
        initial_evidence=initial_code,
        pathology=predictions[0].get("pathology") if predictions else None,
        top_k_features=EXPLAIN_TOP_K_FEATURES,
    )
    similar_cases = semantic_search_case(age=age, sex=sex, evidences=codes, initial_evidence=initial_code, top_k=5)

    return jsonify({
        "resolved_evidence": evidence_markdown(codes, initial_code),
        "predictions": predictions,
        "explanation": explanation,
        "similar_cases": similar_cases,
    })


@api.post("/case/from-text")
def case_from_text():
    """Convenience endpoint: infer structured evidence codes from free text first."""
    body = request.get_json(silent=True) or {}
    text = str(body.get("text", "")).strip()
    if not text:
        return _error("`text` is required.")

    inferred = infer_evidence_codes_from_text(text)
    codes = [item["code"] for item in inferred]
    age, sex = parse_age_sex_from_text(text)
    return jsonify({
        "age": age,
        "sex": sex,
        "evidences": codes,
        "evidence_meanings": [item["meaning"] for item in inferred],
        "initial_evidence": select_initial_evidence(inferred),
    })
