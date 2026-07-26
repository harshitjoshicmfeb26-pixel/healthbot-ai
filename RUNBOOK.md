# RUNBOOK

Step-by-step instructions to run, test, and (optionally) retrain HealthBot.

## 1. Set up the environment

```bash
cd healthcare_nlp_assistant
python3 -m venv venv
source venv/bin/activate          # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

No `.env` file is required — every setting in `config.py` has a safe
default. Copy `.env.example` to `.env` only if you want to override
something (a different port, enabling Ollama, etc.).

## 2. Run the app

```bash
python3 server.py
```

You should see:

```
HealthBot — AI Healthcare NLP Assistant
Starting on http://0.0.0.0:5000  (Ctrl+C to stop)
```

Open **http://localhost:5000** in a browser. The chat loads, calls
`/api/chat/start` automatically, and you can start describing symptoms
right away — the repo ships with already-trained model artifacts in
`saved_models/`, so no training step is required to try it.

Try the four example chips under the chat box, or type your own, e.g.:

- `I have fever, headache and body pain since yesterday`
- `mujhe khansi, jee michalna aur seene mein jalan hai` (Hinglish)
- `मला ताप आणि डोकेदुखी आहे` (Marathi)
- `I have severe chest pain and difficulty breathing` (triggers the red-flag banner)

## 3. Run the test suite

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -v
```

60+ tests cover every `utils/` module, the explainability function, the
session store, and the Flask API (via Flask's test client — no live server
needed for the tests).

## 4. Exercise the API directly (curl)

```bash
# Health & metadata
curl http://localhost:5000/api/health
curl http://localhost:5000/api/meta

# Start a session
curl -X POST http://localhost:5000/api/chat/start -H "Content-Type: application/json" -d '{}'
# -> {"session_id": "...", "reply": "Hello! I am HealthBot ..."}

# Send a message (reuse the session_id from above)
curl -X POST http://localhost:5000/api/chat/message -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "message": "I have a cough, fever and sore throat for 2 days"}'

# Structured prediction (bypasses the conversational flow)
curl -X POST http://localhost:5000/api/case/predict -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": "M", "evidences": "shortness of breath, wheezing, cough", "initial_evidence": "shortness of breath", "top_n": 5}'

# Stateless one-shot analysis
curl -X POST http://localhost:5000/api/analyze -H "Content-Type: application/json" \
  -d '{"text": "49 year old female with cough, nausea, heartburn and stomach pain"}'
```

## 5. Using the real DDXPlus dataset (optional)

The bundled `data/release_evidences.json` and `data/release_conditions.json`
are a **small, hand-built demo subset** (12 of 49 pathologies, ~25 of 223
evidence codes) — see `data/README_DEMO_DATA.md`. They exist so the project
runs and the test suite passes without any download. The model artifacts
already in `saved_models/` were trained on the **full** dataset, so
predictions already work for anything that maps to evidence the model
recognizes; what's missing without the real files is just the friendly
question text for codes outside the demo subset, and severity ratings for
pathologies outside the 12 included.

To use the real data:

1. Download `release_evidences.json`, `release_conditions.json`,
   `train.csv`, `validate.csv`, and `test.csv` from
   <https://huggingface.co/datasets/aai530-group6/ddxplus>.
2. Replace the two JSON files in `data/` with the official ones.
3. Place the three CSVs in the same `data/` folder.
4. **Confirm the severity-scale polarity once**, before trusting any
   severity-based triage message:
   ```bash
   python3 -m utils.severity_engine
   ```
   This prints severity values for conditions that are unambiguously
   high- and low-acuity. `utils/severity_engine.py`'s docstring explains
   exactly what to look for and how to flip `SEVERE_END` if needed.
5. (Optional — the shipped `saved_models/*.pkl` already work) Retrain:
   ```bash
   python3 model/train.py
   ```
   This re-fits the TF-IDF vectorizer + classifier comparison (Naive Bayes
   vs. linear SGD, plus LightGBM if `USE_LIGHTGBM_CANDIDATE=True` and
   `pip install -r requirements-optional.txt` has been run), rebuilds the
   similarity-search index, and writes fresh `classification_report.csv` /
   `confusion_matrix.png` / `test_metrics.csv` to `saved_models/`.

Run training as a module if you ever see `ModuleNotFoundError` for an
internal import:

```bash
python3 -m model.train
```

## 6. Optional extras

```bash
pip install -r requirements-optional.txt
```

Then in `.env` (copy from `.env.example` if you haven't already):

```bash
BIOBERT_ENABLED=True                  # semantic symptom matching
OLLAMA_ENABLED=True                   # requires a local Ollama install
USE_OLLAMA_RESPONSE_FORMATTER=True
OLLAMA_MODEL=qwen2.5:7b                # any model you've already `ollama pull`ed
OLLAMA_NLU_ENABLED=True                # optional JSON symptom/slot extractor, not a diagnosis engine
OLLAMA_NLU_MODEL=qwen3.5:9b             # best local choice here for Hindi/Marathi/Hinglish extraction
USE_LIGHTGBM_CANDIDATE=True            # adds LightGBM to model/train.py's comparison
```

Recommended local LLM split:

- `OLLAMA_NLU_MODEL=qwen3.5:9b` for multilingual symptom/slot extraction.
- `OLLAMA_MODEL=medgemma:4b` or `qwen3.5:9b` only for final response wording.
- Keep the supervised classifier as the only disease prediction engine.

## 7. Production-style serving

The Flask dev server (`python3 server.py`) is fine for local use and demos.
For anything closer to production:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 "server:app"
```

Keep `-w 1` (a single worker) unless you swap the in-memory session store
(`chatbot/session_store.py`) for a shared backend like Redis — see that
module's docstring for why multiple workers would otherwise lose track of
mid-conversation sessions.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'model'` when running a script directly | Run as a module from the project root, e.g. `python3 -m model.train`, or run `python3 server.py` (which already sets up `sys.path` correctly for the whole app). |
| Predictions seem random / very low confidence everywhere | Expected for evidence codes that don't form a coherent clinical picture (the demo data uses real evidence codes but you're free to mix any). Try one of the example chips for a clean signal. |
| `AttributeError: Can't get attribute ... on <module 'sklearn...'>` when loading `saved_models/*.pkl` | A scikit-learn version mismatch. Reinstall the exact pinned version: `pip install scikit-learn==1.5.2`. |
| Port 5000 already in use | `FLASK_PORT=5050 python3 server.py`, or set `FLASK_PORT` in `.env`. |
