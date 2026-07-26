# Changelog — Gradio-to-Flask refactor

This round of changes replaces the Gradio UI with a Flask REST API and a
hand-built HTML/CSS/JS frontend, adds dependency-free model explainability,
and trims the project to what actually ships. It follows an earlier
refactor (NLTK/spaCy removal, negation detection, severity engine, CI) —
see the "Carried over" section below for what that round already fixed.

## Added

- **`api/routes.py`** — A Flask Blueprint exposing the chatbot, the
  structured predictor, and metadata as a clean JSON API
  (`/api/chat/*`, `/api/case/*`, `/api/analyze`, `/api/meta`, `/api/health`).
- **`chatbot/session_store.py`** — Thread-safe in-memory store bridging
  stateless HTTP requests to the stateful, multi-turn `ChatSession` object,
  with TTL-based expiry.
- **`model/predictor.py::explain_case()`** — Token-level, exact (not
  approximated) explanation of a prediction, computed directly from the
  linear classifier's coefficients × TF-IDF weights. No SHAP/LIME
  dependency. See `docs/ARCHITECTURE.md` for the full reasoning and the
  fail-closed behavior if a non-linear model is ever selected.
- **`chatbot/bot.py::assessment_explanation()` / `greeting_message()`** —
  small public bridges connecting the chat assessment pipeline to the new
  explainability function and to the API layer, without exposing the
  module's private constants.
- **`static/`** — A chat-first HTML/CSS/JS frontend (no build step) with a
  live "case insight" panel: detected language, evidence understood, the
  current differential with confidence bars, severity triage, and an
  expandable token-level explanation.
- **`requirements-optional.txt`** — BioBERT (torch + transformers),
  scikit-fuzzy, and LightGBM moved out of the base install; each is gated
  behind its own `.env` flag and every module that uses one fails closed if
  it's missing.
- **LightGBM candidate** in `model/train.py`, gated by
  `USE_LIGHTGBM_CANDIDATE`, off by default.
- New tests: `tests/test_predictor_explain.py`, `tests/test_session_store.py`,
  `tests/test_api.py` (Flask test-client based, no live server needed).

## Removed

- **Gradio entirely** — `app.py`, `run_app.ps1`,
  `healthcare_nlp_assistant_usage_guide.ipynb`, and the Gradio/markupsafe
  dependencies. The 3-tab Gradio UI (Clinical Case Predictor / Disease
  Predictor / Semantic Search / HealthBot Chat) is now one chat surface
  backed by the same underlying pipeline, plus `/api/case/predict` for
  structured input if you want it without the conversational flow.
- **One-off audit/evaluation scripts** whose findings are now permanent,
  tested features rather than standalone reports:
  `model/evaluate_evidence_bridge.py`, `model/evaluate_multilingual.py`,
  `model/audit_classifier.py`, `model/run_regression_suite.py`,
  `model/capture_chatbot_transcripts.py`, and the `audits/` folder of their
  output. The calibration notes they produced (severity-scale polarity,
  negation edge cases) are preserved in `docs/ARCHITECTURE.md` instead of
  living in CSVs nobody re-reads.
- **Redundant docs**: `AUDIT_SOURCE_TABLE.md`, `PROJECT_FLOW_ARCHITECTURE.md`,
  `project_overview.md` consolidated into `README.md` +
  `docs/ARCHITECTURE.md`.

## Changed

- `config.py` — dropped all Gradio settings, added `FLASK_HOST` /
  `FLASK_PORT` / `FLASK_DEBUG` / `SESSION_TTL_MINUTES` /
  `EXPLAIN_TOP_K_FEATURES`.
- `requirements.txt` — now only what's needed to run the chatbot + ML
  predictor + Flask server: Flask, scikit-learn, numpy, pandas, joblib,
  scipy, python-dotenv, matplotlib (training-time confusion matrix only),
  requests (Ollama HTTP client).
- `docs/generate_architecture_diagram.py` — rewritten as a dependency-free
  pure-Python SVG generator (no matplotlib/graphviz needed) reflecting the
  new Flask + HTML/CSS/JS architecture.

## Carried over from the previous (pre-Flask) refactor

For context — these were already in place and are unchanged by this round:

- NLTK/spaCy removed in favor of regex/dict-based NLP (`utils/negation.py`,
  `utils/multilingual_normalizer.py`).
- NegEx-style negation detection, a severity-aware triage engine reading
  the official DDXPlus per-pathology severity field, a FastAPI-style pytest
  suite, and a GitHub Actions CI workflow.
- Fixes for a threading deadlock, alias-matching failures, and contraction
  negation handling found during live testing.
