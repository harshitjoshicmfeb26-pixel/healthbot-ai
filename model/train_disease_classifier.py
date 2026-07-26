"""
Train the simplified supervised disease classifier.

This trainer intentionally does not use `case_text` or
`differential_diagnosis` as model input because the generated simplified
`case_text` includes the target disease name and the differential list. Using
those fields would leak the answer and produce fake confidence.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    SIMPLIFIED_DISEASE_METADATA_PATH,
    SIMPLIFIED_DISEASE_MODEL_PATH,
    SIMPLIFIED_TRAIN_DATA_PATH,
    SIMPLIFIED_TRAIN_ROWS,
)
from model.predict_disease import build_model_input


INPUT_COLUMNS = [
    "age",
    "gender",
    "symptoms_text",
    "pain_location",
    "previous_disease_or_history",
    "genetic_or_family_history",
]
TARGET_COLUMN = "disease"


def load_training_data() -> pd.DataFrame:
    if not Path(SIMPLIFIED_TRAIN_DATA_PATH).exists():
        raise FileNotFoundError(f"Simplified training data not found: {SIMPLIFIED_TRAIN_DATA_PATH}")
    nrows = SIMPLIFIED_TRAIN_ROWS if SIMPLIFIED_TRAIN_ROWS > 0 else None
    df = pd.read_csv(SIMPLIFIED_TRAIN_DATA_PATH, nrows=nrows)
    missing = set(INPUT_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Simplified training data is missing columns: {sorted(missing)}")
    return df.fillna("unknown")


def add_model_input(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()

    def _pain_intensity(symptoms_text: str) -> str:
        match = re.search(r"\bpain intensity:\s*([0-9]|10)\b", str(symptoms_text or ""), flags=re.I)
        return f"pain intensity: {match.group(1)}" if match else "unknown"

    prepared["model_input"] = [
        build_model_input(
            age=row["age"],
            gender=row["gender"],
            symptoms_text=row["symptoms_text"],
            severity=_pain_intensity(row["symptoms_text"]),
            pain_location=row["pain_location"],
            previous_disease_or_history=row["previous_disease_or_history"],
            genetic_or_family_history=row["genetic_or_family_history"],
        )
        for _, row in prepared.iterrows()
    ]
    return prepared


def main() -> None:
    df = add_model_input(load_training_data())
    print(f"Loaded simplified training rows: {len(df):,}")
    print(f"Diseases/classes: {df[TARGET_COLUMN].nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["model_input"],
        df[TARGET_COLUMN].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET_COLUMN].astype(str),
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
    ])

    print("Training TF-IDF + Logistic Regression disease classifier ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    print(classification_report(y_test, y_pred, zero_division=0))

    Path(SIMPLIFIED_DISEASE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SIMPLIFIED_DISEASE_MODEL_PATH, compress=3)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": "tfidf_logistic_regression",
        "train_rows_used": int(len(df)),
        "input_columns": INPUT_COLUMNS,
        "target_column": TARGET_COLUMN,
        "leakage_excluded_columns": ["case_text", "differential_diagnosis", "severity"],
        "metrics": metrics,
        "classes": sorted(df[TARGET_COLUMN].astype(str).unique().tolist()),
    }
    Path(SIMPLIFIED_DISEASE_METADATA_PATH).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model saved at: {SIMPLIFIED_DISEASE_MODEL_PATH}")


if __name__ == "__main__":
    main()
