"""Training and evaluation entry point."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from spam_detector.data import load_dataset
from spam_detector.model import build_pipeline

DEFAULT_MODEL_PATH = Path("models/spam_classifier.joblib")
DEFAULT_METRICS_PATH = Path("models/metrics.json")


@dataclass
class TrainingResult:
    classifier: str
    dataset: str
    n_train: int
    n_test: int
    threshold: float
    metrics: dict = field(default_factory=dict)
    cross_val_f1: dict = field(default_factory=dict)
    trained_at: str = ""
    model_path: str = ""


def _scores(pipeline, texts) -> np.ndarray:
    return pipeline.predict_proba(list(texts))[:, 1]


def choose_threshold(y_true, scores, target_precision: float | None) -> float:
    """Pick the decision threshold.

    Defaults to 0.5. With ``target_precision`` we instead take the lowest
    threshold that still reaches that precision, which maximises recall while
    keeping the number of misfiled legitimate messages under control.
    """
    if target_precision is None:
        return 0.5
    precision, _recall, thresholds = precision_recall_curve(y_true, scores)
    eligible = [
        float(threshold)
        for threshold, prec in zip(thresholds, precision[:-1])
        if prec >= target_precision
    ]
    return min(eligible) if eligible else 0.5


def evaluate(y_true, scores, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=["ham", "spam"], zero_division=0, output_dict=True
        ),
    }


MIN_PER_CLASS = 4


def _check_dataset(frame: pd.DataFrame, dataset: str) -> None:
    counts = frame["label"].value_counts()
    missing = {0, 1} - set(counts.index)
    if missing:
        names = ", ".join("spam" if label == 1 else "ham" for label in sorted(missing))
        raise ValueError(f"{dataset} contains no {names} examples; both classes are required.")
    if counts.min() < MIN_PER_CLASS:
        raise ValueError(
            f"{dataset} has only {counts.min()} example(s) of the rarest class; "
            f"at least {MIN_PER_CLASS} per class are needed to train and evaluate."
        )


def _fold_counts(y_train: pd.Series, cv_folds: int) -> tuple[int, int]:
    """Fit the requested cross-validation to the data.

    Returns the usable number of evaluation folds (0 disables cross-validation) and the
    number of folds the probability calibrator may use, so that tiny datasets train
    instead of failing deep inside scikit-learn.
    """
    smallest_class = int(y_train.value_counts().min())
    evaluation_folds = min(cv_folds, smallest_class) if smallest_class >= 4 else 0
    calibration_folds = 5 if smallest_class >= 25 else 2
    return evaluation_folds, calibration_folds


def train(
    dataset: str = "sms-spam",
    classifier: str = "linear-svm",
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metrics_path: Path | str | None = DEFAULT_METRICS_PATH,
    test_size: float = 0.2,
    target_precision: float | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
) -> TrainingResult:
    """Train the pipeline on ``dataset`` and persist the fitted model."""
    frame: pd.DataFrame = load_dataset(dataset)
    _check_dataset(frame, dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["label"],
        test_size=test_size,
        stratify=frame["label"],
        random_state=random_state,
    )

    if int(y_train.value_counts().min()) < 2:
        raise ValueError(
            f"test_size={test_size} leaves fewer than 2 training examples for one class; "
            "lower test_size or provide more data."
        )

    cv_folds, calibration_folds = _fold_counts(y_train, cv_folds)
    pipeline = build_pipeline(classifier, calibration_folds=calibration_folds)
    cv_summary: dict = {}
    if cv_folds > 1:
        folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(pipeline, x_train, y_train, cv=folds, scoring="f1")
        cv_summary = {"mean": float(cv_scores.mean()), "std": float(cv_scores.std())}

    pipeline.fit(x_train, y_train)
    scores = _scores(pipeline, x_test)
    threshold = choose_threshold(y_test, scores, target_precision)
    metrics = evaluate(y_test, scores, threshold)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    result = TrainingResult(
        classifier=classifier,
        dataset=dataset,
        n_train=int(len(x_train)),
        n_test=int(len(x_test)),
        threshold=float(threshold),
        metrics=metrics,
        cross_val_f1=cv_summary,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model_path=str(model_path),
    )
    joblib.dump(
        {"pipeline": pipeline, "threshold": float(threshold), "metadata": asdict(result)},
        model_path,
    )

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


def format_summary(result: TrainingResult) -> str:
    metrics = result.metrics
    lines = [
        f"classifier      : {result.classifier}",
        f"dataset         : {result.dataset} ({result.n_train} train / {result.n_test} test)",
        f"threshold       : {result.threshold:.3f}",
        f"accuracy        : {metrics['accuracy']:.4f}",
        f"precision (spam): {metrics['precision']:.4f}",
        f"recall (spam)   : {metrics['recall']:.4f}",
        f"f1 (spam)       : {metrics['f1']:.4f}",
        f"roc auc         : {metrics['roc_auc']:.4f}",
        f"confusion matrix: {metrics['confusion_matrix']}",
    ]
    if result.cross_val_f1:
        lines.append(
            f"cv f1           : {result.cross_val_f1['mean']:.4f} "
            f"(+/- {result.cross_val_f1['std']:.4f})"
        )
    lines.append(f"saved model     : {result.model_path}")
    return "\n".join(lines)
