import pytest

from spam_detector.cli import main
from spam_detector.train import train

ROWS = [
    ("win a free prize now", "spam"),
    ("claim your free cash prize", "spam"),
    ("click here for free money", "spam"),
    ("you won a free holiday", "spam"),
    ("urgent claim your reward", "spam"),
    ("call me when you land", "ham"),
    ("lunch at noon works for me", "ham"),
    ("see you at the meeting", "ham"),
    ("the train is running late", "ham"),
    ("thanks for sending the report", "ham"),
]
SPAM_ONLY = [row for row in ROWS if row[1] == "spam"]


def _write_csv(path, rows):
    path.write_text("text,label\n" + "".join(f"{text},{label}\n" for text, label in rows))
    return path


def test_tiny_dataset_trains_without_cross_validation(tmp_path):
    dataset = _write_csv(tmp_path / "tiny.csv", ROWS[:4] + ROWS[5:9])
    result = train(
        dataset=str(dataset),
        model_path=tmp_path / "model.joblib",
        metrics_path=None,
        test_size=0.2,
    )
    assert result.cross_val_f1 == {}
    assert (tmp_path / "model.joblib").exists()


def test_single_class_dataset_is_rejected(tmp_path):
    dataset = _write_csv(tmp_path / "spam_only.csv", SPAM_ONLY)
    with pytest.raises(ValueError, match="contains no ham examples"):
        train(dataset=str(dataset), model_path=tmp_path / "model.joblib", metrics_path=None)


def test_cli_reports_dataset_errors_without_traceback(tmp_path, capsys):
    dataset = _write_csv(tmp_path / "one_each.csv", [ROWS[0], ROWS[5]])
    exit_code = main(
        ["train", "--dataset", str(dataset), "--model-path", str(tmp_path / "m.joblib")]
    )
    assert exit_code == 2
    assert "error: " in capsys.readouterr().err
