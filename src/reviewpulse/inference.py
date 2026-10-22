"""Real inference using locally trusted ReviewPulse sklearn joblib pipelines."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from .config import EMOTION_MODEL_PATH, SENTIMENT_MODEL_PATH
from .preprocessing import clean_review


@dataclass(frozen=True)
class Prediction:
    review: str
    sentiment: str
    emotion: str
    sentiment_confidence: float | None
    emotion_confidence: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Copy trained model from Kaggle into models/.")
    # joblib uses pickle. Load ONLY artifacts produced by your own trusted training.
    model = joblib.load(path)
    if not callable(getattr(model, "predict", None)):
        raise TypeError(f"Model at {path} does not provide predict().")
    return model


@lru_cache(maxsize=1)
def load_models() -> tuple[Any, Any]:
    return _load(SENTIMENT_MODEL_PATH), _load(EMOTION_MODEL_PATH)


def _confidence(model: Any, inputs: Sequence[str], predictions: np.ndarray) -> list[float | None]:
    """Only probabilities are confidence; SVC decision scores are NOT probabilities."""
    if not callable(getattr(model, "predict_proba", None)):
        return [None] * len(inputs)
    probs = np.asarray(model.predict_proba(inputs), dtype=float)
    classes = list(getattr(model, "classes_", []))
    result = []
    for i, predicted in enumerate(predictions):
        try:
            p = float(probs[i, classes.index(predicted)])
            result.append(round(p, 4) if np.isfinite(p) else None)
        except (ValueError, IndexError, TypeError):
            result.append(None)
    return result


def predict_reviews(texts: Sequence[str], *, sentiment_model=None, emotion_model=None) -> pd.DataFrame:
    """Batch inference with original fitted vectorizers; never refit on user reviews."""
    if sentiment_model is None or emotion_model is None:
        loaded_s, loaded_e = load_models()
        sentiment_model = sentiment_model or loaded_s
        emotion_model = emotion_model or loaded_e

    original = list(texts)
    cleaned = [clean_review(v) for v in original]
    valid_ids = [i for i, value in enumerate(cleaned) if value]
    rows = [dict(review=t, sentiment=None, emotion=None,
                 sentiment_confidence=None, emotion_confidence=None, status="empty")
            for t in cleaned]
    if valid_ids:
        batches = [cleaned[i] for i in valid_ids]
        sent = np.asarray(sentiment_model.predict(batches))
        emot = np.asarray(emotion_model.predict(batches))
        sent_conf = _confidence(sentiment_model, batches, sent)
        emot_conf = _confidence(emotion_model, batches, emot)
        for j, idx in enumerate(valid_ids):
            rows[idx].update(sentiment=str(sent[j]), emotion=str(emot[j]),
                             sentiment_confidence=sent_conf[j],
                             emotion_confidence=emot_conf[j], status="ok")
    return pd.DataFrame(rows)


def predict_review(text: str) -> dict:
    if not clean_review(text):
        raise ValueError("Please provide a non-empty review.")
    return predict_reviews([text]).iloc[0].to_dict()
