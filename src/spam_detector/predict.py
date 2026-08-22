"""Inference helpers around a trained model artifact."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path("models/spam_classifier.joblib")


@dataclass(frozen=True)
class Prediction:
    text: str
    label: str
    spam_probability: float

    @property
    def is_spam(self) -> bool:
        return self.label == "spam"


class SpamDetector:
    """Loads a persisted pipeline and scores messages with the trained threshold."""

    def __init__(self, pipeline, threshold: float = 0.5, metadata: dict | None = None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.metadata = metadata or {}

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_MODEL_PATH) -> SpamDetector:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one first: `spam-detector train`."
            )
        artifact = joblib.load(model_path)
        return cls(artifact["pipeline"], artifact.get("threshold", 0.5), artifact.get("metadata"))

    def score(self, texts: Sequence[str]) -> list[float]:
        """Return the spam probability for each message."""
        return [float(score) for score in self.pipeline.predict_proba(list(texts))[:, 1]]

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        return [
            Prediction(text, "spam" if score >= self.threshold else "ham", score)
            for text, score in zip(texts, self.score(texts))
        ]

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]
