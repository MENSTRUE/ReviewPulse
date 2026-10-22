from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "reviewpulse_sentiment_tfidf.joblib"
EMOTION_MODEL_PATH = MODEL_DIR / "reviewpulse_emotion_tfidf.joblib"
MAX_REVIEW_CHARS = 5000
