import json

import numpy as np

from spam_detector.predict import SpamDetector
from spam_detector.train import choose_threshold

SPAM = "WINNER!! Claim your FREE $1000 prize now at http://claim-now.example"
HAM = "Can you review my pull request when you get a chance? It is a small refactor."


def test_training_writes_model_and_metrics(trained_model):
    model_path, result = trained_model
    assert model_path.exists()
    metrics = json.loads((model_path.parent / "metrics.json").read_text())
    assert metrics["classifier"] == "logreg"
    assert result.metrics["accuracy"] >= 0.8


def test_detector_separates_obvious_spam_from_ham(trained_model):
    model_path, _ = trained_model
    detector = SpamDetector.load(model_path)
    spam, ham = detector.predict([SPAM, HAM])
    assert spam.is_spam
    assert not ham.is_spam
    assert spam.spam_probability > ham.spam_probability


def test_choose_threshold_defaults_to_half():
    assert choose_threshold([0, 1], np.array([0.1, 0.9]), None) == 0.5


def test_choose_threshold_respects_target_precision():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.7, 0.9])
    threshold = choose_threshold(y_true, scores, target_precision=1.0)
    predicted = (scores >= threshold).astype(int)
    assert not (predicted[y_true == 0]).any()
