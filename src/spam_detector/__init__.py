"""Spam email detection: dataset loading, model training, inference and serving."""

from spam_detector.model import build_pipeline
from spam_detector.predict import SpamDetector

__all__ = ["SpamDetector", "build_pipeline", "__version__"]

__version__ = "0.1.0"
