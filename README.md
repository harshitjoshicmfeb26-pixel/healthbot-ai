# HealthBot — Structured Clinical NLP Assistant

A conversational symptom checker built over a **structured clinical-evidence
machine learning model**, with multilingual NLP, negation-aware red-flag
triage, official-severity-based safety checks, and direct linear-model feature
contributions for transparent inspection.

This is the second major refactor of the project. The first removed NLTK/
spaCy and added negation detection, a severity engine, and CI-backed pytest
coverage (see `docs/CHANGELOG.md`). **This round removes Gradio** and
replaces it with a small Flask REST API and a hand-built HTML/CSS/JS chat
frontend, adds dependency-free model explainability, and trims the codebase
to what actually ships.

![Architecture diagram](docs/architecture_diagram.svg)

## Why this design

| Decision | Reasoning |
|---|---|
| **Structured evidence → TF-IDF → linear classifier, rather than raw-text deep learning** | DDXPlus is fundamentally structured: age, sex, an initial presenting evidence, and a closed vocabulary of evidence codes. Free-text interpretation happens upstream; the classifier receives structured evidence tokens. TF-IDF provides a compact sparse representation suited to this downstream task and keeps the final linear model directly interpretable. See `model/predictor.py::explain_case()`. |
| **Deterministic negation, temporal handling and red-flag rules** | Safety-sensitive phrases such as “no chest pain”, “cannot be ruled out”, and “I had no chest pain yesterday but today it started” need predictable, testable handling. The project uses scoped negation, contraction and pseudo-negation rules, narrow temporal reassertion handling, and a separately tested red-flag path. Controlled benchmark results are not claims of real-world correctness or clinical validation. |
| **Optional LLM components are not required for the canonical runtime** | The default path uses deterministic extraction, the structured DDXPlus classifier, and deterministic response templates. Optional Ollama NLU can extract symptoms/slots, and optional Ollama formatting can rewrite structured output; both are disabled by default, grounded, and never serve as the pathology prediction engine. |
| **Flask + plain HTML/CSS/JavaScript** | Flask serves the REST API and lightweight frontend without a separate frontend build system. The API remains a clean, inspectable surface that another client could use in the future. |
| **Direct linear-model explainability** | For the deployed linear model, a feature's contribution to a class decision score is derived directly as `tfidf_weight(token) × coefficient(token, class)`. This is an exact contribution to the model's score, not a claim of medical causality or the exact reason a patient has a condition. |

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
server.py                  Flask entrypoint — serves REST API + static frontend
config.py                  Central settings; reads optional .env configuration

api/routes.py              REST API: health, metadata, chat, text analysis,
                           evidence extraction, structured prediction

chatbot/
  bot.py                   ChatSession — the multi-turn conversational state machine
  session_store.py         In-memory session store bridging stateless HTTP to ChatSession

model/
  predictor.py             Loads core structured-model artifacts; prediction,
                           explanation, and optional similar-case search
  train.py                 Structured DDXPlus training/evaluation pipeline;
                           compares MultinomialNB, SGDClassifier, and optional LightGBM

utils/
  language_detector.py       English / Hinglish / Marathi / Romanized Marathi detection
  multilingual_normalizer.py Deterministic symptom normalization to canonical concepts
  medical_synonyms.py         Curated multilingual aliases + validated typo mappings
  fuzzy_symptom_matcher.py    Conservative fuzzy normalization candidates; not automatically
                             promoted into classifier evidence
  negation.py                 Scoped negation, contractions, pseudo-negation, and narrow
                             temporal reassertion handling
  red_flag_rules.py           Negation-aware emergency/urgent red-flag rules
  severity_engine.py          DDXPlus condition severity as an additional safety signal
  ddxplus_decoder.py          Text ↔ DDXPlus evidence mapping, contextual precedence,
                             polarity handling, and INITIAL_EVIDENCE selection
  clinical_case_features.py   Structured DDXPlus case → ML feature-token representation
  response_summarizer.py      Deterministic response templates
  ollama_client.py            Optional grounded Ollama response formatting; disabled by default
  biobert_embedder.py         Optional semantic fallback/matching; disabled by default

static/                    Frontend — index.html, css/style.css, js/app.js (no build step)
data/                      Tracked runtime metadata; CSV splits are training/evaluation-only
saved_models/              Core LFS artifacts plus ignored optional search/experimental artifacts
evaluation/                Controlled model-comparison and evidence-benchmark tooling/results
tests/                     pytest suite — 170 tests at the current verified release checkpoint
docs/                      Architecture, training-to-prediction flow, interview guide,
                           changelog, and final architecture diagram
```

## API reference

All endpoints are mounted under `/api`. Full request/response examples are in
`RUNBOOK.md`.

| Method & path | Purpose |
|---|---|
| `GET  /api/health` | Liveness check |
| `GET  /api/meta` | Model metadata: selected algorithm, pathology list, and optional-component status |
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
`simplified_disease_classifier.pkl` is retained for controlled offline
comparison only and is not a production fallback.

## Retraining on the synthetic DDXPlus dataset

The repo ships with **already-trained** core model artifacts (`saved_models/`),
trained on the official synthetic DDXPlus dataset. The current artifact
metadata records 49 pathology classes and 200,001 sampled training rows,
49,999 validation rows, and 49,999 test rows. You don't need to retrain to use
the app, and the large training CSVs are not required for inference. If you
obtain `data/train.csv`, `validate.csv`, and `test.csv` from the official
DDXPlus source and want to retrain, run:

```bash
python3 model/train.py
```

See `RUNBOOK.md` for the full data-setup steps and the optional severity-scale
inspection (`utils/severity_engine.describe_severity_scale()`).

## Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m pytest tests/ -v
```

## Optional extras

Nothing in `requirements.txt` requires these — see `requirements-optional.txt`
and `.env.example` for how to turn each one on:

- **BioBERT semantic fallback/matching** — optional and disabled by default
  (`utils/biobert_embedder.py`).
- **Local Ollama NLU and response formatting** — optional, disabled by default,
  and grounded; neither component predicts pathology (`utils/ollama_nlu_extractor.py`,
  `utils/ollama_client.py`).
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
