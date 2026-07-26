"""
Optional BioBERT semantic matcher.

BioBERT is used here as an encoder, not as a diagnosis generator. When enabled,
it embeds the user's symptom sentence and compares it with official DDXPlus
evidence aliases. The structured TF-IDF classifier still performs the final
PATHOLOGY prediction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterable

import numpy as np

from config import (
    BIOBERT_ALIAS_INDEX_PATH,
    BIOBERT_ENABLED,
    BIOBERT_LOCAL_FILES_ONLY,
    BIOBERT_MIN_SIMILARITY,
    BIOBERT_MODEL_NAME,
    BIOBERT_TOP_K_ALIASES,
)


TOKEN_RE = r"[a-zA-Z]{3,}"
GENERIC_MATCH_TOKENS = {
    "about",
    "after",
    "also",
    "before",
    "blockage",
    "consultation",
    "discomfort",
    "feel",
    "feeling",
    "general",
    "have",
    "keeping",
    "new",
    "person",
    "related",
    "some",
    "today",
    "trouble",
    "vague",
    "well",
    "with",
    "your",
}
MEDICAL_TOKEN_GROUPS = (
    {"urine", "urinary", "urinate", "urinating", "urination", "pee", "peeing", "bladder", "urethra"},
    {"cough", "breath", "breathing", "wheeze", "wheezing", "sputum", "phlegm", "lung", "chest"},
    {"chest", "heart", "cardiac", "pressure", "angina", "palpitation"},
    {"stomach", "abdominal", "abdomen", "belly", "nausea", "vomit", "diarrhea", "reflux", "heartburn"},
    {"head", "headache", "migraine", "dizzy", "dizziness", "vertigo"},
    {"fever", "temperature", "chills", "sweating", "sweat"},
    {"skin", "rash", "itch", "itching", "swelling", "lesion"},
    {"joint", "muscle", "back", "neck", "knee", "leg", "arm", "pain"},
    {"throat", "swallow", "swallowing", "voice", "hoarse"},
)


def biobert_status(check_dependencies: bool = False) -> dict:
    """Return lightweight status for debug panels and docs."""
    if not BIOBERT_ENABLED:
        return {
            "enabled": False,
            "available": False,
            "model_name": BIOBERT_MODEL_NAME,
            "reason": "Disabled by BIOBERT_ENABLED=False",
        }

    if not check_dependencies:
        return {
            "enabled": True,
            "available": None,
            "model_name": BIOBERT_MODEL_NAME,
            "reason": "Enabled; dependencies load only when semantic fallback is used",
        }

    try:
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer  # noqa: F401
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "model_name": BIOBERT_MODEL_NAME,
            "reason": f"Missing optional dependency: {exc}",
        }

    return {
        "enabled": True,
        "available": True,
        "model_name": BIOBERT_MODEL_NAME,
        "reason": "Ready; model weights load on first semantic match",
    }


@lru_cache(maxsize=1)
def _load_biobert():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BIOBERT_MODEL_NAME,
        local_files_only=BIOBERT_LOCAL_FILES_ONLY,
    )
    model = AutoModel.from_pretrained(
        BIOBERT_MODEL_NAME,
        local_files_only=BIOBERT_LOCAL_FILES_ONLY,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device, torch


def _mean_pool(last_hidden_state, attention_mask, torch):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def encode_texts(texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
    """Encode text with BioBERT and return L2-normalized sentence vectors."""
    tokenizer, model, device, torch = _load_biobert()
    clean_texts = [str(text or "").strip() for text in texts]
    if not clean_texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = []
    for start in range(0, len(clean_texts), batch_size):
        batch = clean_texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"], torch)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    return _normalize_vectors(np.vstack(vectors))


@lru_cache(maxsize=4)
def _encoded_alias_index(alias_items: tuple[tuple[str, tuple[str, ...]], ...]) -> np.ndarray:
    aliases = [alias for alias, _ in alias_items]
    return encode_texts(aliases)


def _alias_codes_json(alias_items: tuple[tuple[str, tuple[str, ...]], ...]) -> list[str]:
    return [json.dumps(codes) for _, codes in alias_items]


def _load_disk_alias_index(alias_items: tuple[tuple[str, tuple[str, ...]], ...]) -> np.ndarray | None:
    if not BIOBERT_ALIAS_INDEX_PATH.exists():
        return None
    try:
        cached = np.load(BIOBERT_ALIAS_INDEX_PATH, allow_pickle=False)
        aliases = cached["aliases"].astype(str).tolist()
        codes_json = cached["codes_json"].astype(str).tolist()
        model_name = str(cached["model_name"])
        vectors = cached["vectors"].astype(np.float32)
    except Exception:
        return None

    expected_aliases = [alias for alias, _ in alias_items]
    if model_name != BIOBERT_MODEL_NAME:
        return None
    if aliases != expected_aliases:
        return None
    if codes_json != _alias_codes_json(alias_items):
        return None
    return vectors


def _save_disk_alias_index(
    alias_items: tuple[tuple[str, tuple[str, ...]], ...],
    vectors: np.ndarray,
) -> None:
    try:
        BIOBERT_ALIAS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            BIOBERT_ALIAS_INDEX_PATH,
            model_name=np.array(BIOBERT_MODEL_NAME),
            aliases=np.array([alias for alias, _ in alias_items]),
            codes_json=np.array(_alias_codes_json(alias_items)),
            vectors=vectors.astype(np.float32),
        )
    except Exception:
        return


@lru_cache(maxsize=2)
def _full_alias_index(alias_items: tuple[tuple[str, tuple[str, ...]], ...]) -> np.ndarray:
    vectors = _load_disk_alias_index(alias_items)
    if vectors is not None:
        return vectors
    vectors = _encoded_alias_index(alias_items)
    _save_disk_alias_index(alias_items, vectors)
    return vectors


def _tokens(text: str) -> set[str]:
    import re

    return {token.lower() for token in re.findall(TOKEN_RE, str(text or ""))}


def _is_relevant_candidate(query: str, alias: str) -> bool:
    query_tokens = _tokens(query)
    alias_tokens = _tokens(alias)
    if not query_tokens or not alias_tokens:
        return False

    shared = {
        token
        for token in query_tokens & alias_tokens
        if len(token) >= 4 and token not in GENERIC_MATCH_TOKENS
    }
    if shared:
        return True

    for group in MEDICAL_TOKEN_GROUPS:
        if query_tokens & group and alias_tokens & group:
            return True
    return False


def semantic_alias_matches(
    text: str,
    alias_map: dict[str, list[str]],
    exclude_codes: set[str] | None = None,
    min_similarity: float | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """
    Return BioBERT nearest alias matches for a symptom sentence.

    This function fails closed: if BioBERT is disabled, not installed, or model
    weights cannot load, it returns an empty list so the app falls back to exact
    DDXPlus alias matching.
    """
    if not BIOBERT_ENABLED or not str(text or "").strip():
        return []

    status = biobert_status(check_dependencies=True)
    if not status["available"]:
        return []

    threshold = BIOBERT_MIN_SIMILARITY if min_similarity is None else float(min_similarity)
    limit = BIOBERT_TOP_K_ALIASES if top_k is None else int(top_k)
    excluded = {str(code).upper() for code in (exclude_codes or set())}

    alias_items = tuple(
        (str(alias), tuple(str(code).upper() for code in codes))
        for alias, codes in sorted(alias_map.items())
        if alias and codes
    )
    if not alias_items:
        return []

    try:
        query_vector = encode_texts([text])
        alias_vectors = _full_alias_index(alias_items)
    except Exception:
        return []

    if query_vector.size == 0 or alias_vectors.size == 0:
        return []

    scores = alias_vectors @ query_vector[0]
    relevant_indices = [
        idx
        for idx, (alias, _) in enumerate(alias_items)
        if _is_relevant_candidate(text, alias)
    ]
    if not relevant_indices:
        return []
    ranked = sorted(relevant_indices, key=lambda idx: scores[idx], reverse=True)
    results = []
    seen_codes = set(excluded)

    for idx in ranked:
        score = float(scores[idx])
        if score < threshold:
            break
        alias, codes = alias_items[int(idx)]
        new_codes = [code for code in codes if code not in seen_codes]
        if not new_codes:
            continue
        for code in new_codes:
            seen_codes.add(code)
        results.append({
            "alias": alias,
            "codes": new_codes,
            "similarity": round(score, 4),
            "model": BIOBERT_MODEL_NAME,
        })
        if len(results) >= limit:
            break

    return results
