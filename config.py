"""
config.py — Central configuration loader
Reads .env and exposes typed settings across the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() == "true"


# ── App ───────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "HealthBot — AI Healthcare NLP Assistant")

# ── Model artifacts & data ──────────────────────────────────────
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "saved_models/disease_classifier.pkl")
VECTORIZER_PATH = BASE_DIR / os.getenv("VECTORIZER_PATH", "saved_models/tfidf_vectorizer.pkl")
ENCODER_PATH = BASE_DIR / os.getenv("ENCODER_PATH", "saved_models/label_encoder.pkl")
EMBEDDINGS_PATH = BASE_DIR / os.getenv("EMBEDDINGS_PATH", "saved_models/tfidf_matrix.pkl")
TRAIN_DATA_PATH = BASE_DIR / os.getenv("TRAIN_DATA_PATH", "data/train.csv")
TEST_DATA_PATH = BASE_DIR / os.getenv("TEST_DATA_PATH", "data/test.csv")
VALIDATE_DATA_PATH = BASE_DIR / os.getenv("VALIDATE_DATA_PATH", "data/validate.csv")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
TRAIN_SAMPLE_ROWS = int(os.getenv("TRAIN_SAMPLE_ROWS", "200000"))
SEARCH_INDEX_ROWS = int(os.getenv("SEARCH_INDEX_ROWS", "100000"))
TRAIN_RANDOM_STATE = int(os.getenv("TRAIN_RANDOM_STATE", "42"))
MODEL_METADATA_PATH = BASE_DIR / os.getenv("MODEL_METADATA_PATH", "saved_models/model_metadata.json")
SEARCH_CASES_PATH = BASE_DIR / os.getenv("SEARCH_CASES_PATH", "saved_models/search_cases.pkl")
USE_LIGHTGBM_CANDIDATE = _bool("USE_LIGHTGBM_CANDIDATE", "False")

# Supervised non-RAG text classifier over the simplified DDXPlus dataset.
SIMPLIFIED_TRAIN_DATA_PATH = BASE_DIR / os.getenv("SIMPLIFIED_TRAIN_DATA_PATH", "data/simplified_train.csv")
SIMPLIFIED_DISEASE_MODEL_PATH = BASE_DIR / os.getenv(
    "SIMPLIFIED_DISEASE_MODEL_PATH",
    "saved_models/simplified_disease_classifier.pkl",
)
SIMPLIFIED_DISEASE_METADATA_PATH = BASE_DIR / os.getenv(
    "SIMPLIFIED_DISEASE_METADATA_PATH",
    "saved_models/simplified_disease_classifier_metadata.json",
)
SIMPLIFIED_TRAIN_ROWS = int(os.getenv("SIMPLIFIED_TRAIN_ROWS", "100000"))

# ── Flask server ──────────────────────────────────────────────
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = _bool("FLASK_DEBUG", "False")

# ── Local Ollama response formatting (optional) ────────────────
OLLAMA_ENABLED = _bool("OLLAMA_ENABLED", "False")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
USE_OLLAMA_RESPONSE_FORMATTER = _bool("USE_OLLAMA_RESPONSE_FORMATTER", "False")

# Optional local Ollama NLU extractor. This is not a diagnosis engine: it only
# extracts symptoms/slots as JSON for multilingual text that the deterministic
# synonym maps did not fully understand.
OLLAMA_NLU_ENABLED = _bool("OLLAMA_NLU_ENABLED", "False")
OLLAMA_NLU_MODEL = os.getenv("OLLAMA_NLU_MODEL", "qwen3.5:9b")
OLLAMA_NLU_TIMEOUT = int(os.getenv("OLLAMA_NLU_TIMEOUT", "20"))

# ── Optional BioBERT semantic symptom matcher (heavy, off by default) ──
BIOBERT_ENABLED = _bool("BIOBERT_ENABLED", "False")
BIOBERT_MODEL_NAME = os.getenv("BIOBERT_MODEL_NAME", "dmis-lab/biobert-base-cased-v1.1")
BIOBERT_MIN_SIMILARITY = float(os.getenv("BIOBERT_MIN_SIMILARITY", "0.84"))
BIOBERT_TOP_K_ALIASES = int(os.getenv("BIOBERT_TOP_K_ALIASES", "2"))
BIOBERT_LOCAL_FILES_ONLY = _bool("BIOBERT_LOCAL_FILES_ONLY", "True")
BIOBERT_ALIAS_INDEX_PATH = BASE_DIR / os.getenv(
    "BIOBERT_ALIAS_INDEX_PATH",
    "saved_models/biobert_alias_index.npz",
)

# ── NLP / chatbot behavior ──────────────────────────────────────
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
MAX_CLARIFICATION_TURNS = int(os.getenv("MAX_CLARIFICATION_TURNS", "3"))
RED_FLAG_FIRST = _bool("RED_FLAG_FIRST", "True")
SHOW_DEBUG_OUTPUT = _bool("SHOW_DEBUG_OUTPUT", "True")
EXPLAIN_TOP_K_FEATURES = int(os.getenv("EXPLAIN_TOP_K_FEATURES", "6"))

# ── Session store ────────────────────────────────────────────────
# In-memory chat sessions time out after this many minutes of inactivity.
# Fine for a single-process local/demo deployment; swap SessionStore's
# backing dict for Redis if you need multi-process or persistent sessions.
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "60"))
