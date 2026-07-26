"""
Structured clinical evidence inference engine.

Primary API:
  predict_case(age, sex, evidences, initial_evidence, top_n)
  semantic_search_case(age, sex, evidences, initial_evidence, top_k)

Backward-compatible API:
  predict(text, top_n)
  semantic_search(query, top_k)

The current dataset is structured, not free-text. The UI maps natural symptom
phrases to official DDXPlus evidence values before calling this module.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    EMBEDDINGS_PATH,
    ENCODER_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SEARCH_CASES_PATH,
    TOP_K_RESULTS,
    VECTORIZER_PATH,
)
from utils.clinical_case_features import (  # noqa: E402
    EVIDENCE_CODE_RE,
    build_case_feature_text,
    evidence_codes,
    format_case_record,
)
from utils.ddxplus_decoder import decode_condition, decode_evidence  # noqa: E402


DATASET_MODE = "structured_clinical_evidence_v1"

_vectorizer = None
_clf = None
_le = None
_X_search = None
_search_cases = None
_metadata = None


def _score_flag(score: float) -> str:
    if score >= 0.60:
        return "HIGH"
    if score >= CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _predict_probabilities(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[0]
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        scores = scores[0] if getattr(scores, "ndim", 1) > 1 else scores
        scores = np.asarray(scores, dtype=float)
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()
    raise AttributeError("Classifier does not expose predict_proba or decision_function.")


def _metadata_is_current() -> bool:
    if not MODEL_METADATA_PATH.exists():
        return False
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("dataset_mode") == DATASET_MODE


def _ensure_loaded():
    global _vectorizer, _clf, _le, _X_search, _search_cases, _metadata

    if _clf is not None:
        return

    required = [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH, EMBEDDINGS_PATH, SEARCH_CASES_PATH]
    if not all(Path(path).exists() for path in required) or not _metadata_is_current():
        print("No current structured clinical model found. Training now ...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "train.py")], check=True)

    _vectorizer = joblib.load(VECTORIZER_PATH)
    _clf = joblib.load(MODEL_PATH)
    _le = joblib.load(ENCODER_PATH)
    _X_search = joblib.load(EMBEDDINGS_PATH)
    _search_cases = joblib.load(SEARCH_CASES_PATH)
    _metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def model_metadata() -> dict:
    _ensure_loaded()
    return dict(_metadata or {})


def known_pathology_names() -> List[str]:
    """
    The full closed set of pathology display names the classifier was
    trained on (all 49 DDXPlus classes, not just the current top-k).

    Used by `utils.ollama_client` to verify that a generated explanation
    never names a disease outside this set.
    """
    _ensure_loaded()
    names = []
    for pathology in _le.classes_:
        decoded = decode_condition(str(pathology))
        names.append(decoded.get("display_name", str(pathology)))
    return names


def _prediction_rows(feature_text: str, top_n: int, source: str = "structured ML") -> List[Dict]:
    _ensure_loaded()
    if not feature_text.strip():
        return [{
            "rank": 1,
            "disease": "No valid clinical evidence provided",
            "label_id": -1,
            "confidence": 0.0,
            "confidence_pct": "0.0%",
            "flag": "ERROR",
            "source": source,
            "score_type": "model confidence",
        }]

    X = _vectorizer.transform([feature_text])
    proba = _predict_probabilities(_clf, X)
    top_cols = np.argsort(proba)[::-1][:top_n]
    sorted_scores = np.sort(proba)[::-1]
    top_margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

    rows = []
    for rank, col_idx in enumerate(top_cols, 1):
        score = float(proba[col_idx])
        encoded_label = _clf.classes_[col_idx]
        pathology = str(_le.inverse_transform([encoded_label])[0])
        decoded_condition = decode_condition(pathology)
        display_name = decoded_condition.get("display_name", pathology)
        rows.append({
            "rank": rank,
            "disease": display_name,
            "pathology": pathology,
            "label_id": int(encoded_label),
            "confidence": round(score, 4),
            "confidence_pct": f"{score * 100:.1f}%",
            "flag": _score_flag(score),
            "source": source,
            "score_type": "model confidence",
            "top_margin": round(top_margin, 4) if rank == 1 else None,
            "top_margin_pct": f"{top_margin * 100:.1f}%" if rank == 1 else None,
        })
    return rows


def predict_case(
    age: Any,
    sex: Any,
    evidences: Any,
    initial_evidence: Any = "",
    top_n: int = 5,
) -> List[Dict]:
    """Predict PATHOLOGY from structured clinical evidence fields."""
    feature_text = build_case_feature_text(
        age=age,
        sex=sex,
        evidences=evidences,
        initial_evidence=initial_evidence,
    )
    return _prediction_rows(feature_text, top_n=int(top_n), source="structured clinical evidence")


# ─────────────────────────────────────────────────────────────
# Lightweight, dependency-free explainability
# ─────────────────────────────────────────────────────────────
# Why not SHAP/LIME: the production model is TF-IDF features feeding a
# *linear* classifier (logistic-loss SGD, selected by model/train.py's own
# top5_accuracy comparison — see saved_models/model_comparison.csv). A linear
# model is already exactly interpretable: each feature's contribution to a
# class score is just `tfidf_weight(feature) * coefficient(feature, class)`,
# summed over features. SHAP/LIME approximate that relationship for opaque
# models with extra sampling and a heavy dependency; for a linear model they
# would just reconstruct the same number this module computes directly and
# exactly. If a future non-linear candidate (e.g. LightGBM, see
# model/train.py) is ever selected instead, `explain_case` fails closed —
# see the `hasattr(_clf, "coef_")` branch below — rather than silently
# returning a wrong or misleading explanation.

_EXPLAINABLE_PREFIXES = ("ev_", "sex_", "initial_", "age_")
_MANGLED_VALUE_CODE_RE = re.compile(r"^(E_\d+)_(V_\d+|\d+)$")


def _reconstruct_evidence_code(remainder: str) -> str:
    """
    Recover the original 'E_NN_@_V_NN' evidence code from the TF-IDF token's
    mangled form. `_code_token()` (utils/clinical_case_features.py) collapses
    the literal '_@_' separator into a single underscore when building
    vocabulary tokens, e.g. 'E_55_@_V_187' -> 'E_55_V_187'. That's fine as a
    training feature, but it is not a valid evidence code on its own, so
    decode_evidence() would otherwise report it as unknown. This reverses
    that one specific collision for display purposes only.
    """
    match = _MANGLED_VALUE_CODE_RE.match(remainder)
    if match:
        return f"{match.group(1)}_@_{match.group(2)}"
    return remainder


def _humanize_feature_token(token: str) -> str:
    """Translate one internal TF-IDF feature token back into plain English."""
    if token.startswith("ev_"):
        code = _reconstruct_evidence_code(token[len("ev_"):])
        return decode_evidence(code).get("meaning", token)
    if token.startswith("initial_"):
        code = _reconstruct_evidence_code(token[len("initial_"):])
        meaning = decode_evidence(code).get("meaning", token)
        return f"Initial complaint: {meaning}"
    if token.startswith("sex_"):
        return {"sex_F": "Sex: female", "sex_M": "Sex: male"}.get(token, "Sex: unspecified")
    if token.startswith("age_decade_"):
        return f"Age in the {token[len('age_decade_'):]}s"
    if token.startswith("age_"):
        return f"Age group: {token[len('age_'):].replace('_', ' ')}"
    return token


def explain_case(
    age: Any,
    sex: Any,
    evidences: Any,
    initial_evidence: Any = "",
    pathology: str | None = None,
    top_k_features: int = 6,
) -> Dict:
    """
    Explain why the model leaned toward one pathology for one structured case.

    If `pathology` is omitted, explains the top-ranked prediction for this
    case. Returns:
        {
          "pathology": "...",
          "contributions": [{"feature", "meaning", "weight"}, ...],
          "note": "" | a reason the list is empty,
        }
    `contributions` is sorted by how strongly each token pushed the score
    toward `pathology`, most influential first. An empty list with a
    non-empty `note` means explanation is unavailable for this case/model,
    not that no evidence mattered.
    """
    _ensure_loaded()
    feature_text = build_case_feature_text(
        age=age, sex=sex, evidences=evidences, initial_evidence=initial_evidence
    )
    if not feature_text.strip():
        return {"pathology": pathology, "contributions": [], "note": "No structured evidence to explain."}

    X = _vectorizer.transform([feature_text])

    if pathology:
        try:
            encoded_label = _le.transform([str(pathology)])[0]
            class_idx = int(np.where(_clf.classes_ == encoded_label)[0][0])
        except (ValueError, IndexError):
            pathology = None

    if not pathology:
        proba = _predict_probabilities(_clf, X)
        class_idx = int(np.argmax(proba))

    resolved_pathology = str(_le.inverse_transform([_clf.classes_[class_idx]])[0])

    if not hasattr(_clf, "coef_"):
        return {
            "pathology": resolved_pathology,
            "contributions": [],
            "note": (
                "The currently selected model does not expose linear "
                "coefficients, so token-level explanation is unavailable."
            ),
        }

    contributions = X.toarray()[0] * _clf.coef_[class_idx]
    feature_names = _vectorizer.get_feature_names_out()

    ranked = [
        idx
        for idx in np.argsort(contributions)[::-1]
        if contributions[idx] > 0
        and " " not in feature_names[idx]
        and feature_names[idx].startswith(_EXPLAINABLE_PREFIXES)
        and not feature_names[idx].startswith("initial_in_evidence_")
    ]

    rows = [
        {
            "feature": feature_names[idx],
            "meaning": _humanize_feature_token(feature_names[idx]),
            "weight": round(float(contributions[idx]), 4),
        }
        for idx in ranked[: max(1, int(top_k_features))]
    ]
    note = "" if rows else "No symptom evidence pushed this case toward this pathology."
    return {"pathology": resolved_pathology, "contributions": rows, "note": note}


def _text_to_feature_text(text: str) -> str:
    """Convert pasted evidence-code text into model feature text."""
    raw = str(text or "")
    codes = evidence_codes(raw)
    if codes:
        age_match = re.search(r"\bage\s*[:=]?\s*(\d{1,3})\b", raw, flags=re.I)
        sex_match = re.search(r"\bsex\s*[:=]?\s*([MF])\b", raw, flags=re.I)
        initial_match = re.search(r"\binitial(?:_evidence)?\s*[:=]?\s*(E_\d+(?:_@_(?:V_)?\d+)?)\b", raw, flags=re.I)
        return build_case_feature_text(
            age=age_match.group(1) if age_match else None,
            sex=sex_match.group(1) if sex_match else None,
            evidences=codes,
            initial_evidence=initial_match.group(1) if initial_match else "",
        )

    # Reached only when the text contains something code-LIKE (it passed the
    # EVIDENCE_CODE_RE guard in predict() below) but no individual token
    # fully validates as a real evidence code (e.g. "E_53xyz"). There used
    # to be a fallback here that ran plain prose through an nltk
    # tokenizer/lemmatizer and fed the result to this model as `text_*`
    # tokens — but the trained TF-IDF vocabulary contains only `ev_*` /
    # `sex_*` / `age_*` tokens, so those `text_*` features could never match
    # anything and the "prediction" was guaranteed to be noise. Returning an
    # empty string here is the honest behavior: `_prediction_rows()` already
    # turns an empty feature string into a clear "no valid evidence" row.
    return ""


def predict(text: str, top_n: int = 5) -> List[Dict]:
    """Backward-compatible predictor for evidence-code text."""
    if not EVIDENCE_CODE_RE.search(str(text or "")):
        return [{
            "rank": 1,
            "disease": "Recognized symptom evidence required",
            "pathology": "Recognized symptom evidence required",
            "label_id": -1,
            "confidence": 0.0,
            "confidence_pct": "0.0%",
            "flag": "INFO",
            "source": "structured clinical evidence",
            "score_type": "not run",
            "warning": "The current model needs symptoms that can be mapped to official DDXPlus evidence metadata.",
        }]
    feature_text = _text_to_feature_text(text)
    return _prediction_rows(feature_text, top_n=int(top_n), source="structured evidence text")


def _similarity_rows(feature_text: str, top_k: int) -> List[Dict]:
    from sklearn.metrics.pairwise import cosine_similarity

    _ensure_loaded()
    if not feature_text.strip():
        return []

    q_vec = _vectorizer.transform([feature_text])
    sims = cosine_similarity(q_vec, _X_search)[0]
    top_idx = np.argsort(sims)[::-1][:max(top_k * 10, top_k)]

    rows = []
    seen = set()
    for idx in top_idx:
        score = float(sims[idx])
        if score <= 0:
            continue
        case = dict(_search_cases[int(idx)])
        key = (
            case.get("pathology"),
            case.get("age"),
            case.get("sex"),
            case.get("initial_evidence"),
            tuple(case.get("evidences", [])[:8]),
        )
        if key in seen:
            continue
        seen.add(key)
        decoded_condition = decode_condition(case.get("pathology", "Unknown pathology"))
        display_name = decoded_condition.get("display_name", case.get("pathology", "Unknown pathology"))
        rows.append({
            "rank": len(rows) + 1,
            "disease": display_name,
            "pathology": case.get("pathology", "Unknown pathology"),
            "similarity": round(score, 4),
            "similarity_pct": f"{score * 100:.1f}%",
            "age": case.get("age"),
            "sex": case.get("sex"),
            "initial_evidence": case.get("initial_evidence"),
            "evidence_count": case.get("evidence_count"),
            "evidences": case.get("evidences", []),
            "differential_diagnosis": case.get("differential_diagnosis", []),
            "text": format_case_record(case),
        })
        if len(rows) >= top_k:
            break
    return rows


def semantic_search_case(
    age: Any,
    sex: Any,
    evidences: Any,
    initial_evidence: Any = "",
    top_k: int | None = None,
) -> List[Dict]:
    feature_text = build_case_feature_text(
        age=age,
        sex=sex,
        evidences=evidences,
        initial_evidence=initial_evidence,
    )
    return _similarity_rows(feature_text, int(top_k or TOP_K_RESULTS))


def semantic_search(query: str, top_k: int = None) -> List[Dict]:
    """Backward-compatible search for pasted evidence-code text."""
    if not EVIDENCE_CODE_RE.search(str(query or "")):
        return []
    feature_text = _text_to_feature_text(query)
    return _similarity_rows(feature_text, int(top_k or TOP_K_RESULTS))
