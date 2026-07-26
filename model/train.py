"""
Train the structured clinical evidence model.

New dataset schema:
AGE, SEX, DIFFERENTIAL_DIAGNOSIS, PATHOLOGY, EVIDENCES, INITIAL_EVIDENCE

The model predicts PATHOLOGY from AGE + SEX + INITIAL_EVIDENCE + EVIDENCES.
DIFFERENTIAL_DIAGNOSIS is not used as an input feature because it already
contains disease candidates and would leak the answer. It is saved only as
reference metadata for similar-case display.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    BASE_DIR,
    EMBEDDINGS_PATH,
    ENCODER_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SEARCH_CASES_PATH,
    SEARCH_INDEX_ROWS,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_RANDOM_STATE,
    TRAIN_SAMPLE_ROWS,
    USE_LIGHTGBM_CANDIDATE,
    VALIDATE_DATA_PATH,
    VECTORIZER_PATH,
)
from utils.clinical_case_features import (  # noqa: E402
    REQUIRED_COLUMNS,
    row_to_case_record,
    row_to_feature_text,
)


DATASET_MODE = "structured_clinical_evidence_v1"
MODEL_SELECTION_SORT = ["top3_accuracy", "macro_f1", "top1_accuracy"]
EXPLAINABLE_MODEL_TOLERANCE = 0.002


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} dataset not found: {path}")
    print(f"Loading {split_name}: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} is missing columns: {sorted(missing)}")
    print(f"    {split_name}: {df.shape[0]:,} rows | {df['PATHOLOGY'].nunique()} pathologies")
    return df


def stratified_sample(df: pd.DataFrame, max_rows: int, split_name: str) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()

    print(f"Sampling {split_name} to {max_rows:,} rows with pathology stratification ...")
    sampled = (
        df.groupby("PATHOLOGY", group_keys=False)
        .sample(frac=max_rows / len(df), random_state=TRAIN_RANDOM_STATE)
    )
    missing_labels = set(df["PATHOLOGY"].unique()) - set(sampled["PATHOLOGY"].unique())
    if missing_labels:
        backfill = df[df["PATHOLOGY"].isin(missing_labels)].groupby("PATHOLOGY", group_keys=False).head(1)
        sampled = pd.concat([sampled, backfill], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=TRAIN_RANDOM_STATE).reset_index(drop=True)


def prepare_features(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    print(f"Building structured feature text for {split_name} ...")
    prepared = df.copy()
    prepared["case_text"] = [row_to_feature_text(row) for _, row in prepared.iterrows()]
    prepared = prepared[prepared["case_text"].str.strip() != ""]
    return prepared


def fit_label_encoder(train_df: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(train_df["PATHOLOGY"].astype(str))
    print(f"Train pathologies/classes: {len(encoder.classes_)}")
    return encoder


def filter_seen_labels(df: pd.DataFrame, encoder: LabelEncoder, split_name: str) -> pd.DataFrame:
    seen = set(encoder.classes_)
    mask = df["PATHOLOGY"].astype(str).isin(seen)
    excluded = int((~mask).sum())
    if excluded:
        print(f"Excluded {excluded:,} {split_name} rows with unseen pathologies.")
    return df[mask].copy()


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[A-Za-z0-9_@/+-]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )


def candidate_models() -> Dict[str, object]:
    models: Dict[str, object] = {
        "multinomial_nb": MultinomialNB(alpha=0.05),
        "sgd_log_loss": SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.05,
            class_weight="balanced",
            max_iter=60,
            tol=1e-3,
            random_state=TRAIN_RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if USE_LIGHTGBM_CANDIDATE:
        try:
            from lightgbm import LGBMClassifier

            # Note for model/predictor.py's explain_case(): LightGBM does not
            # expose `coef_`, so if this candidate wins the comparison below,
            # the live token-level explanation in the UI degrades to a clear
            # "explanation unavailable for this model" note rather than a
            # silently wrong one — see explain_case()'s docstring.
            models["lightgbm"] = LGBMClassifier(
                n_estimators=250,
                num_leaves=63,
                learning_rate=0.08,
                class_weight="balanced",
                random_state=TRAIN_RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            print(
                "USE_LIGHTGBM_CANDIDATE=True but `lightgbm` is not installed "
                "(pip install -r requirements-optional.txt). Skipping that candidate."
            )

    return models


def probability_scores(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    if hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    raise AttributeError("Model has no probability or decision score method.")


def top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top_cols = np.argsort(proba, axis=1)[:, -k:]
    top_labels = classes[top_cols]
    return float(np.mean([true in row for true, row in zip(y_true, top_labels)]))


def evaluate_model(name: str, clf, X, y_true: np.ndarray) -> dict:
    y_pred = clf.predict(X)
    proba = probability_scores(clf, X)
    top1 = accuracy_score(y_true, y_pred)
    return {
        "model": name,
        "accuracy": top1,
        "top1_accuracy": top1,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": top_k_accuracy(y_true, proba, clf.classes_, 3),
        "top5_accuracy": top_k_accuracy(y_true, proba, clf.classes_, 5),
        "explainable": bool(hasattr(clf, "coef_")),
    }


def select_model_from_comparison(comparison: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    """
    Select a deployment model for the chat UI.

    Top-5 is almost saturated on this dataset, so it is not useful as the
    primary chooser. Prefer top-3, macro F1, and top-1 accuracy. If an
    explainable linear model is very close to the best raw metric winner,
    keep the explainable model because the UI depends on exact coefficient
    explanations.
    """
    ordered = comparison.sort_values(MODEL_SELECTION_SORT, ascending=False).reset_index(drop=True)
    best = ordered.iloc[0]
    explainable = ordered[
        (ordered["explainable"])
        & (ordered["top3_accuracy"] >= best["top3_accuracy"] - EXPLAINABLE_MODEL_TOLERANCE)
        & (ordered["macro_f1"] >= best["macro_f1"] - EXPLAINABLE_MODEL_TOLERANCE)
    ]
    if not explainable.empty:
        selected = explainable.iloc[0]
        reason = (
            "Selected the best explainable model within tolerance of the raw metric winner "
            f"(sort={MODEL_SELECTION_SORT}, tolerance={EXPLAINABLE_MODEL_TOLERANCE})."
        )
    else:
        selected = best
        reason = f"Selected the best raw metric winner by {MODEL_SELECTION_SORT}."

    comparison = comparison.copy()
    comparison["selected_for_deployment"] = comparison["model"] == selected["model"]
    comparison["selection_reason"] = reason
    return str(selected["model"]), reason, comparison


def train_and_select(train_df: pd.DataFrame, validate_df: pd.DataFrame):
    vectorizer = build_vectorizer()
    print("Vectorizing structured clinical cases ...")
    X_train = vectorizer.fit_transform(train_df["case_text"])
    X_validate = vectorizer.transform(validate_df["case_text"])
    y_train = train_df["label_enc"].values
    y_validate = validate_df["label_enc"].values

    rows = []
    fitted = {}
    for name, model in candidate_models().items():
        print(f"\nTraining candidate: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        metrics = evaluate_model(name, model, X_validate, y_validate)
        metrics["training_time_sec"] = round(elapsed, 2)
        rows.append(metrics)
        fitted[name] = model
        print(
            f"    acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"top3={metrics['top3_accuracy']:.4f} "
            f"top5={metrics['top5_accuracy']:.4f} "
            f"time={elapsed:.1f}s"
        )

    comparison = pd.DataFrame(rows)
    best_name, selection_reason, comparison = select_model_from_comparison(comparison)
    print(f"\nSelected model: {best_name}")
    print(selection_reason)
    return vectorizer, fitted[best_name], X_train, best_name, comparison, selection_reason


def save_reports(best_name: str, clf, encoder: LabelEncoder, vectorizer, test_df: pd.DataFrame, out_dir: Path) -> None:
    print("\nEvaluating selected model on test split ...")
    X_test = vectorizer.transform(test_df["case_text"])
    y_test = test_df["label_enc"].values
    y_pred = clf.predict(X_test)
    proba = probability_scores(clf, X_test)

    final_metrics = {
        "model": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": top_k_accuracy(y_test, proba, clf.classes_, 3),
        "top5_accuracy": top_k_accuracy(y_test, proba, clf.classes_, 5),
    }
    pd.DataFrame([final_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)
    print("Test metrics:", json.dumps(final_metrics, indent=2))

    target_names = encoder.inverse_transform(np.arange(len(encoder.classes_)))
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    save_confusion_matrix(y_test, y_pred, encoder, out_dir / "confusion_matrix.png")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, encoder: LabelEncoder, output_path: Path) -> None:
    top_labels = pd.Series(y_true).value_counts().head(30).index.to_numpy()
    mask = np.isin(y_true, top_labels)
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=top_labels)
    names = encoder.inverse_transform(top_labels)

    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        ylabel="True pathology",
        xlabel="Predicted pathology",
        title="Confusion Matrix - Top 30 Pathologies",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_search_index(train_df: pd.DataFrame, vectorizer) -> tuple:
    search_df = stratified_sample(train_df, SEARCH_INDEX_ROWS, "search index")
    print(f"Building search index with {len(search_df):,} cases ...")
    search_matrix = vectorizer.transform(search_df["case_text"])
    search_cases = [row_to_case_record(row) for _, row in search_df.iterrows()]
    return search_matrix, search_cases


def save_artifacts(vectorizer, clf, encoder, search_matrix, search_cases, comparison, metadata: dict) -> None:
    out_dir = Path(MODEL_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    joblib.dump(clf, MODEL_PATH, compress=3)
    joblib.dump(encoder, ENCODER_PATH, compress=3)
    joblib.dump(search_matrix, EMBEDDINGS_PATH, compress=3)
    joblib.dump(search_cases, SEARCH_CASES_PATH, compress=3)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nArtifacts saved to {out_dir}")
    for path in [VECTORIZER_PATH, MODEL_PATH, ENCODER_PATH, EMBEDDINGS_PATH, SEARCH_CASES_PATH, MODEL_METADATA_PATH]:
        print(f"    {Path(path).name}")


def main() -> None:
    train_raw = load_split(TRAIN_DATA_PATH, "train")
    validate_raw = load_split(VALIDATE_DATA_PATH, "validate") if VALIDATE_DATA_PATH.exists() else None
    test_raw = load_split(TEST_DATA_PATH, "test")

    train_raw = stratified_sample(train_raw, TRAIN_SAMPLE_ROWS, "train")
    if validate_raw is None:
        validate_raw = stratified_sample(test_raw, min(50000, len(test_raw)), "validate-from-test")
    elif TRAIN_SAMPLE_ROWS > 0:
        validate_raw = stratified_sample(validate_raw, min(50000, len(validate_raw)), "validate")
    if TRAIN_SAMPLE_ROWS > 0:
        test_raw = stratified_sample(test_raw, min(50000, len(test_raw)), "test")

    train_df = prepare_features(train_raw, "train")
    validate_df = prepare_features(validate_raw, "validate")
    test_df = prepare_features(test_raw, "test")

    encoder = fit_label_encoder(train_df)
    validate_df = filter_seen_labels(validate_df, encoder, "validate")
    test_df = filter_seen_labels(test_df, encoder, "test")

    train_df["label_enc"] = encoder.transform(train_df["PATHOLOGY"].astype(str))
    validate_df["label_enc"] = encoder.transform(validate_df["PATHOLOGY"].astype(str))
    test_df["label_enc"] = encoder.transform(test_df["PATHOLOGY"].astype(str))

    vectorizer, best_clf, X_train, best_name, comparison, selection_reason = train_and_select(train_df, validate_df)
    search_matrix, search_cases = build_search_index(train_df, vectorizer)

    def _portable_path(path) -> str:
        """
        Store a path relative to the project root, not the absolute path on
        whatever machine ran training. The previous version of this script
        wrote `str(TRAIN_DATA_PATH)` directly, which bakes in an
        absolute, machine-specific path (e.g. a Windows user profile
        directory) into a file meant to be shared/committed.
        """
        try:
            return str(Path(path).resolve().relative_to(BASE_DIR.resolve()))
        except ValueError:
            return str(path)  # path is outside BASE_DIR; fall back as-is

    metadata = {
        "dataset_mode": DATASET_MODE,
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "train_data_path": _portable_path(TRAIN_DATA_PATH),
        "validate_data_path": _portable_path(VALIDATE_DATA_PATH),
        "test_data_path": _portable_path(TEST_DATA_PATH),
        "train_rows_used": int(len(train_df)),
        "validate_rows_used": int(len(validate_df)),
        "test_rows_used": int(len(test_df)),
        "search_index_rows": int(len(search_cases)),
        "classes": encoder.classes_.tolist(),
        "selected_model": best_name,
        "model_selection_sort": MODEL_SELECTION_SORT,
        "model_selection_reason": selection_reason,
        "input_features": ["AGE", "SEX", "INITIAL_EVIDENCE", "EVIDENCES"],
        "target": "PATHOLOGY",
        "excluded_input_columns": ["DIFFERENTIAL_DIAGNOSIS"],
    }

    save_artifacts(vectorizer, best_clf, encoder, search_matrix, search_cases, comparison, metadata)
    save_reports(best_name, best_clf, encoder, vectorizer, test_df, Path(MODEL_PATH).parent)
    print("\nTraining complete. Run `python app.py` to launch the UI.")


if __name__ == "__main__":
    main()
