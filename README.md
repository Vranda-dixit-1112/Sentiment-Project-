# Social Media Sentiment Analysis and Prediction System

A B.Tech-level Python project that classifies a short social-media text as **positive**, **negative**, or **neutral** using TF-IDF features and Logistic Regression. The interface is built with Streamlit.

## Features
- Train/test split with stratification
- TF-IDF with unigrams and bigrams
- Logistic Regression classifier
- Accuracy, precision, recall, F1-score, and confusion matrix
- Saved model pipeline and evaluation metrics
- Streamlit prediction page and dataset dashboard

## Requirements
- Python 3.10+
- pip
- Internet access only if you need to install packages; the included demo dataset is local.

## Setup (Windows)
```powershell
cd social_media_sentiment_project
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Setup (macOS/Linux)
```bash
cd social_media_sentiment_project
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset format
`data/social_media_sentiment.csv` must contain:
- `text`: post/comment text
- `sentiment`: one of `positive`, `negative`, `neutral`

Replace the teaching dataset with a larger, representative and properly licensed dataset for meaningful evaluation. Check the data license and attribution requirements. Remove personally identifying information where possible.

## Testing checklist
1. Start with a clearly positive post.
2. Test a clearly negative post.
3. Test a factual/neutral post.
4. Try empty input (the app should ask for text).
5. Try sarcasm, mixed sentiment, emoji-heavy text, abbreviations, and spelling mistakes.
6. Re-run `python train_model.py` after changing the dataset.
7. Confirm the app's dataset dashboard and saved evaluation metrics are updated.

## Important limitations
The included dataset is deliberately small and illustrative. Scores can be misleading on such a dataset and should not be presented as evidence of general performance. A production-quality project needs a larger independent dataset, deduplication, representative language coverage, error analysis, and ideally comparison with other algorithms. TF-IDF + Logistic Regression does not reliably understand sarcasm or context. Model probabilities are not calibrated confidence scores.
