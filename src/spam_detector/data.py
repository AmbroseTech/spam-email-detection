"""Dataset loading utilities.

Two sources are supported:

* the public SMS Spam Collection corpus (downloaded and cached on first use);
* any local CSV that has a text column and a label column.

Both are normalised to a :class:`pandas.DataFrame` with the columns
``text`` (str) and ``label`` (int, 1 = spam, 0 = ham).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_CACHE_DIR = Path("data/raw")
SAMPLE_DATASET = Path(__file__).parent / "datasets" / "sample_spam.csv"

LABEL_ALIASES = {
    "spam": 1,
    "1": 1,
    "true": 1,
    "ham": 0,
    "0": 0,
    "false": 0,
    "legit": 0,
    "not_spam": 0,
}


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Map textual/boolean/numeric labels onto 1 (spam) and 0 (ham)."""
    mapped = labels.astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    unknown = labels[mapped.isna()].unique()
    if len(unknown) > 0:
        raise ValueError(f"Unrecognised label values: {sorted(map(str, unknown))}")
    return mapped.astype(int)


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["text", "label"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    frame = frame.drop_duplicates(subset=["text"])
    return frame.reset_index(drop=True)


def load_csv(
    path: str | Path, text_column: str = "text", label_column: str = "label"
) -> pd.DataFrame:
    """Load a user supplied CSV of labelled messages."""
    frame = pd.read_csv(path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {sorted(missing)}; found {list(frame.columns)}"
        )
    frame = frame[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    frame["label"] = normalize_labels(frame["label"])
    return _clean(frame)


def download_sms_spam(cache_dir: str | Path = DEFAULT_CACHE_DIR, force: bool = False) -> Path:
    """Download the SMS Spam Collection corpus and return the path to the raw file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "SMSSpamCollection"
    if target.exists() and not force:
        return target

    response = requests.get(SMS_SPAM_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("SMSSpamCollection") as handle:
            target.write_bytes(handle.read())
    return target


def load_sms_spam(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Load the SMS Spam Collection corpus, downloading it if it is not cached yet."""
    path = download_sms_spam(cache_dir)
    frame = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="latin-1")
    frame["label"] = normalize_labels(frame["label"])
    return _clean(frame[["text", "label"]])


def load_sample() -> pd.DataFrame:
    """Load the tiny bundled dataset used by the tests and offline smoke runs."""
    return load_csv(SAMPLE_DATASET)


def load_dataset(
    source: str = "sms-spam", cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """Load a dataset by name (``sms-spam``, ``sample``) or from a CSV path."""
    if source == "sms-spam":
        return load_sms_spam(cache_dir)
    if source == "sample":
        return load_sample()
    return load_csv(source)
