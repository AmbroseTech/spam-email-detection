import pandas as pd
import pytest

from spam_detector.data import load_csv, load_sample, normalize_labels


def test_sample_dataset_is_balanced_and_binary():
    frame = load_sample()
    assert set(frame.columns) == {"text", "label"}
    assert set(frame["label"].unique()) == {0, 1}
    assert len(frame) > 40


def test_normalize_labels_accepts_common_spellings():
    labels = normalize_labels(pd.Series(["spam", "HAM", "1", "0", "true"]))
    assert list(labels) == [1, 0, 1, 0, 1]


def test_normalize_labels_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unrecognised label"):
        normalize_labels(pd.Series(["spam", "maybe"]))


def test_load_csv_renames_custom_columns(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("body,is_spam\nbuy now,spam\nsee you soon,ham\n")
    frame = load_csv(path, text_column="body", label_column="is_spam")
    assert list(frame["label"]) == [1, 0]


def test_load_csv_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        load_csv(path)
