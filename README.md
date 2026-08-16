# HealthBot — Structured Clinical NLP Assistant

A conversational symptom checker built over a **structured clinical-evidence
machine learning model**, with multilingual NLP, negation-aware red-flag
triage, official-severity-based safety checks, and a fully transparent,
no-black-box explanation of every prediction.

This is the second major refactor of the project. The first removed NLTK/
spaCy and added negation detection, a severity engine, a FastAPI-style test
suite, and CI (see `docs/CHANGELOG.md`). **This round removes Gradio** and
replaces it with a small Flask REST API and a hand-built HTML/CSS/JS chat
frontend, adds dependency-free model explainability, and trims the codebase
to what actually ships.

![Architecture diagram](docs/architecture_diagram.svg)

## Why this design

| Decision | Reasoning |
|---|---|
| **TF-IDF + linear classifier over structured evidence**, not raw free-text deep learning | DDXPlus is a *structured* dataset (age, sex, a closed vocabulary of ~223 evidence codes). A linear model over those exact features is both more accurate *and* exactly interpretable — every prediction can be explained with real coefficients, not an approximation. See `model/predictor.py::explain_case()`. |
| **Rule-based negation + red-flag triage, not an LLM** | Safety-critical logic (e.g. "no chest pain" must never be miscounted as chest pain) needs to be deterministic and testable. NegEx-style scoped negation (`utils/negation.py`) and a separately-tested red-flag matcher (`utils/red_flag_rules.py`) give 100%-reproducible behavior with full unit-test coverage. |
| **Ollama is a formatter, never the diagnosis engine** | `utils/ollama_client.py` only rewrites already-structured ML + safety output into friendlier prose, and `verify_grounded_response()` rejects any rewrite that invents a disease outside the model's actual top-k or mentions a drug dosage. Disabled by default — the deterministic template formatter (`utils/response_summarizer.py`) is the fallback and cannot hallucinate, because it only renders fields already in the payload. |
| **Flask + plain HTML/CSS/JS, not a SPA framework** | No build step, no node_modules, nothing to explain away in an interview. The API is a clean, inspectable REST surface (`api/routes.py`) that you could point any frontend at. |
| **No SHAP/LIME for explainability** | The production model is linear. `contribution(token) = tfidf_weight(token) × coefficient(token, class)` is the *exact* answer, computed directly — SHAP/LIME would spend a sampling budget reconstructing a number this codebase already has in closed form. See `model/predictor.py::explain_case()` for the full reasoning, and `docs/ARCHITECTURE.md` for how this degrades gracefully if a future non-linear model (e.g. the optional LightGBM candidate) is ever selected. |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py                 # Windows: .venv\Scripts\python.exe server.py
```

Open **http://localhost:5000**. The chatbot, structured ML predictor, and
severity engine all work immediately when Git LFS has materialized the core
model files. The tracked runtime metadata contains the complete DDXPlus
evidence and condition definitions in `data/release_evidences.json` and
`data/release_conditions.json`. Training CSVs are intentionally not bundled
because they are only needed for retraining/evaluation. Full step-by-step
instructions are in [`RUNBOOK.md`](RUNBOOK.md). The architecture walkthrough
is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the training/runtime flow
is [`docs/DATA_TO_TRAINING_TO_PREDICTION_FLOW.md`](docs/DATA_TO_TRAINING_TO_PREDICTION_FLOW.md),
and DDXPlus attribution is in [`data/README.md`](data/README.md).

Git LFS is required for the core `.pkl` files. After installing Git LFS, use
normal clone/pull behavior or run `git lfs pull` from the repository root.
The optional Ollama and BioBERT components are disabled by default and are not
required for deterministic inference.

## Project structure

```
server.py                  Flask entrypoint — serves the API + static frontend
config.py                  Central settings (reads .env, all keys documented in .env.example)

api/routes.py              REST API: chat, structured predict, metadata, health

chatbot/
  bot.py                   ChatSession — the multi-turn conversational state machine
  session_store.py         In-memory session store bridging stateless HTTP to ChatSession

model/
  predictor.py             Loads core artifacts; prediction, explanation, optional semantic search
  train.py                 Retrains from data/*.csv; compares NB / SGD / optional LightGBM

utils/                     NLP building blocks (each independently unit-tested)
  language_detector.py       English / Hinglish / Marathi / Romanized Marathi detection
  multilingual_normalizer.py Symptom phrase -> canonical English terms
  fuzzy_symptom_matcher.py   Fuzzy-match phrasing the dictionary misses
  negation.py                 NegEx-style scoped negation ("no chest pain" != chest pain)
  red_flag_rules.py           Negation-aware emergency/urgent keyword triage
  severity_engine.py          Official DDXPlus per-pathology severity, as a 2nd safety signal
  ddxplus_decoder.py          Evidence-code <-> human-text bridge in both directions
  clinical_case_features.py   Structured-record <-> ML feature-token pipeline
  response_summarizer.py      Deterministic, non-hallucinating response templates
  ollama_client.py            Optional local LLM formatter, with a grounding verifier
  biobert_embedder.py         Optional semantic symptom matching (heavy, off by default)

static/                    Frontend — index.html, css/style.css, js/app.js (no build step)
data/                      Tracked runtime metadata; CSV splits are training/evaluation-only
saved_models/              Core LFS artifacts plus ignored optional search/experimental artifacts
tests/                     pytest suite — 170 tests at the current verified release checkpoint
docs/                      Architecture diagram + its generator script, changelog
```

## API reference

All endpoints are mounted under `/api`. Full request/response examples are in
`RUNBOOK.md`.

| Method & path | Purpose |
|---|---|
| `GET  /api/health` | Liveness check |
| `GET  /api/meta` | Model metadata: selected algorithm, pathology list, BioBERT status |
| `POST /api/chat/start` | Begin a session, get the greeting |
| `POST /api/chat/message` | Send one chat turn; returns the reply *and* a `meta` block (predictions, red flags, severity, language, explanation) |
| `POST /api/chat/reset` | Reset an existing session |
| `POST /api/analyze` | Stateless one-shot analysis of a symptom sentence (no conversational slot-filling) |
| `POST /api/case/predict` | Structured prediction from explicit age/sex/evidence input, plus `explain_case()` |
| `POST /api/case/from-text` | Infer structured evidence codes from free text, without predicting |

Conversational endpoints use an in-memory, process-local session and collect
the presenting complaint and relevant slots before prediction. Sessions reset
when the process restarts; multi-worker deployment requires a shared backend
such as Redis. `/api/analyze` is stateless and analyzes supplied text directly,
so it can behave differently from conversational slot filling.

Core prediction uses the structured DDXPlus classifier. Similar-case search is
optional and uses the ignored `tfidf_matrix.pkl` and `search_cases.pkl`; their
absence does not disable prediction or explanation. The ignored
`simplified_disease_classifier.pkl` is retained for offline comparison only.

## Retraining on the real dataset

The repo ships with **already-trained** core model artifacts (`saved_models/`),
trained on the official DDXPlus dataset (49 pathologies, 200k+ training rows;
see `saved_models/model_metadata.json`). You don't need to retrain to use the
app, and the large training CSVs are not required for inference. If you obtain
`data/train.csv`, `validate.csv`, and `test.csv` from the official DDXPlus
source and want to retrain, run:

```bash
python3 model/train.py
```

See `RUNBOOK.md` for the full data-setup steps, including the one-time
severity-scale calibration check (`utils/severity_engine.describe_severity_scale()`).

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest tests/ -v
```

## Optional extras

Nothing in `requirements.txt` requires these — see `requirements-optional.txt`
and `.env.example` for how to turn each one on:

- **BioBERT semantic matching** — catches symptom phrasing the dictionary +
  fuzzy matcher miss (`utils/biobert_embedder.py`).
- **Local Ollama response formatter** — rewrites the structured output into
  more natural prose, with a grounding verifier (`utils/ollama_client.py`).
- **LightGBM training candidate** — an additional model compared against
  the Naive Bayes / linear baselines during retraining (`model/train.py`).

## Disclaimer

The current verified release checkpoint contains 170 passing tests. Evidence
extraction metrics come from the controlled 96-case benchmark and are not
clinical validation. Classifier metrics are measured on held-out synthetic
DDXPlus data and are not real-patient diagnostic accuracy. The benchmark
checkpoint reports positive precision/recall/F1 of 92.19% / 97.52% / 94.78%,
positive exact-set match of 91.67%, denied precision/recall/F1 of 100% /
82.76% / 90.57%, initial-evidence accuracy of 64/64 (100%), pseudo-negation
100%, red-flag F1 of 100%, temporal exact/F1 of 100% / 100%, and contraction
exact match of 100%.

HealthBot is an educational/portfolio project. It is **not a medical
device** and does not provide a diagnosis. Every response includes this
disclaimer; treat any output as a starting point for a conversation with a
qualified clinician, not a substitute for one.
