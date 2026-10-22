import numpy as np
from reviewpulse.inference import predict_reviews


class FakeEstimator:
    classes_ = np.array(["Negative", "Positive"])

    def predict(self, texts):
        return np.array(["Positive"] * len(texts))

    def predict_proba(self, texts):
        return np.array([[0.2, 0.8]] * len(texts))


class FakeSVC:
    def predict(self, texts):
        return np.array(["Happy"] * len(texts))


def test_batch_empty_rows_and_optional_probabilities():
    result = predict_reviews(["  bagus  ", "  "],
                             sentiment_model=FakeEstimator(), emotion_model=FakeSVC())
    assert result.loc[0,"sentiment"] == "Positive"
    assert result.loc[0,"emotion"] == "Happy"
    assert result.loc[0,"sentiment_confidence"] == 0.8
    assert result.loc[0,"emotion_confidence"] is None
    assert result.loc[1,"status"] == "empty"
