import pytest

from spam_detector.train import train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Train once on the tiny bundled dataset and reuse the artifact across tests."""
    directory = tmp_path_factory.mktemp("model")
    model_path = directory / "spam_classifier.joblib"
    result = train(
        dataset="sample",
        classifier="logreg",
        model_path=model_path,
        metrics_path=directory / "metrics.json",
        cv_folds=3,
    )
    return model_path, result
