"""
Streamlit UI for the social-media sentiment analysis project.
Run: streamlit run app.py
"""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sentiment_pipeline.joblib"
METRICS_PATH = ROOT / "evaluation_metrics.json"
DATA_PATH = ROOT / "data" / "social_media_sentiment.csv"
LABEL_ORDER = ["negative", "neutral", "positive"]

st.set_page_config(page_title="Social Media Sentiment Analyzer", page_icon="💬", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("💬 Social Media Sentiment Analysis & Prediction")
st.caption("B.Tech project • Python • NLP • TF-IDF • Logistic Regression")
st.write(
    "Enter a post, comment, or short review. The model predicts whether its "
    "overall tone is positive, negative, or neutral."
)

if model is None:
    st.warning("The trained model was not found. Run `python train_model.py` from the project folder, then refresh this page.")
    st.stop()

tab_predict, tab_dashboard, tab_about = st.tabs(["🔎 Predict sentiment", "",""])

with tab_predict:
    st.subheader("Analyze a post")
    sample = st.selectbox(
        "Or choose a sample",
        [
            "Choose a sample...",
            "Love the new update, everything works perfectly!",
            "This app keeps crashing and support has not replied.",
            "The event starts at 6 PM on Friday.",
            "These ads are annoying and appear everywhere.",
            "I waited an hour and the event was poorly organized.",
            "This game is full of bugs and feels unfinished.",
            "My order arrived broken and the box was damaged."
        ]
    )
    default_text = "" if sample == "Choose a sample..." else sample
    post = st.text_area(
        "Social media text",
        value=default_text,
        height=130,
        placeholder="Paste a social media post or comment here..."
    )
    if st.button("Analyze text", type="primary"):
        if not post.strip():
            st.error("Please enter some text before analyzing.")
        else:
            predicted = model.predict([post])[0]
            probabilities = model.predict_proba([post])[0]
            probability_df = pd.DataFrame({
                "Sentiment": model.classes_,
                "Probability": probabilities
            }).sort_values("Probability", ascending=False)
            probability = float(probabilities[list(model.classes_).index(predicted)])
            if predicted == "positive":
                st.success(f"Predicted sentiment: POSITIVE  •  model score {probability:.1%}")
            elif predicted == "negative":
                st.error(f"Predicted sentiment: NEGATIVE  •  model score {probability:.1%}")
            else:
                st.info(f"Predicted sentiment: NEUTRAL  •  model score {probability:.1%}")
            st.write("Model class probabilities (not guaranteed confidence):")
            chart_df = probability_df.set_index("Sentiment")
            st.bar_chart(chart_df)
            st.caption("This baseline model may misread sarcasm, slang, emojis, mixed opinions, and context.")

with tab_dashboard:
    st.subheader("Dataset overview")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        col1, col2, col3 = st.columns(3)
        col1.metric("Example posts", len(data))
        col2.metric("Sentiment classes", data["sentiment"].nunique())
        col3.metric("Rows per class (minimum)", int(data["sentiment"].value_counts().min()))
        counts = data["sentiment"].value_counts().reindex(LABEL_ORDER, fill_value=0)
        st.bar_chart(counts)
        st.write("Sample records")
        st.dataframe(data.sample(min(10, len(data)), random_state=42), use_container_width=True)
        st.warning("The included dataset is a small teaching example. Replace it with a larger, appropriately licensed, representative dataset before drawing research conclusions.")
    else:
        st.info("Dataset CSV was not found.")

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.subheader("Latest held-out test evaluation")
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        st.caption("Metrics reflect the small sample dataset, not expected real-world performance.")
        st.json({
            "classification_report": metrics["classification_report"],
            "confusion_matrix_order": metrics["labels"],
            "confusion_matrix": metrics["confusion_matrix"]
        })
    else:
        st.info("Evaluation metrics are not available yet. Run the training script first.")

with tab_about:
    st.subheader("How it works")
    st.markdown("""
    1. **Input:** a social media post or comment.
    2. **TF-IDF:** transforms words and two-word phrases into numeric features.
    3. **Logistic Regression:** predicts one of three labels: negative, neutral, or positive.
    4. **Evaluation:** a stratified held-out test split is used to report accuracy, precision, recall, F1-score, and a confusion matrix.

    **Limitations:** this is a baseline educational system. It does not identify the author's intent or reliably detect sarcasm, irony, language switching, or aspect-specific sentiment. Do not use it to make high-impact decisions about individuals.
    """)
