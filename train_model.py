"""
Train a baseline social-media sentiment classifier.
Run from the project root: python train_model.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "social_media_sentiment.csv"
MODEL_PATH = ROOT / "sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"
LABELS = ["negative", "neutral", "positive"]

def main():
    df = pd.read_csv(DATA_PATH)
    required = {"text", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df = df.dropna(subset=["text", "sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[(df["text"] != "") & df["sentiment"].isin(LABELS)]
    df = df.drop_duplicates(subset=["text"])

    if df["sentiment"].nunique() != 3:
        raise ValueError("Training data must include negative, neutral, and positive labels.")
    if df["sentiment"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for stratified splitting.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.25,
        random_state=42, stratify=df["sentiment"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            ngram_range=(1, 2), min_df=1, max_features=15000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS).tolist()

    print(f"Dataset rows used: {len(df)}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=LABELS, zero_division=0
    ))
    print("Confusion matrix (rows=true, columns=predicted; order: negative, neutral, positive):")
    print(matrix)

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "accuracy": float(accuracy),
            "labels": LABELS,
            "confusion_matrix": matrix,
            "classification_report": report,
            "note": "Metrics are from a small illustrative dataset and are not evidence of real-world performance."
        }, f, indent=2)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
