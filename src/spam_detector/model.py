"""Model definition: normalisation -> word + character TF-IDF -> linear classifier."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from spam_detector.preprocess import normalize_corpus

CLASSIFIERS = ("linear-svm", "logreg", "naive-bayes")


def build_classifier(name: str):
    """Instantiate one of the supported classifiers, all exposing ``predict_proba``."""
    if name == "linear-svm":
        # LinearSVC has the best precision/recall trade-off on short text but no
        # probabilities of its own, so calibrate it to keep the scoring API uniform.
        return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=5)
    if name == "logreg":
        return LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
    if name == "naive-bayes":
        return MultinomialNB(alpha=0.1)
    raise ValueError(f"Unknown classifier {name!r}; choose one of {list(CLASSIFIERS)}")


def build_pipeline(classifier: str = "linear-svm") -> Pipeline:
    """Build the end-to-end pipeline that maps raw message text to a spam score."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_corpus, validate=False)),
            ("features", features),
            ("classifier", build_classifier(classifier)),
        ]
    )
