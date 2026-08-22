"""Text normalisation applied before vectorisation.

Spam messages differ from legitimate mail in *shape* as much as in wording: they
carry links, phone numbers, currency amounts and shouty capitalisation. Replacing
those with stable placeholder tokens lets the vectoriser learn "there is a URL
here" instead of memorising one specific domain.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(
    r"(?:[$£€]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:usd|eur|gbp|dollars?|pounds?)\b)", re.IGNORECASE
)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
NUMBER_RE = re.compile(r"\b\d+\b")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case a message and replace volatile entities with placeholder tokens."""
    text = str(text)
    shouty = " __allcaps__" if _shouty(text) else ""
    text = URL_RE.sub(" __url__ ", text)
    text = EMAIL_RE.sub(" __email__ ", text)
    text = MONEY_RE.sub(" __money__ ", text)
    text = PHONE_RE.sub(" __phone__ ", text)
    text = text.lower()
    text = NUMBER_RE.sub(" __number__ ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text + shouty


def _shouty(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 10:
        return False
    upper = sum(char.isupper() for char in letters)
    return upper / len(letters) > 0.5


def normalize_corpus(texts: Iterable[str]) -> list[str]:
    """Vectoriser-friendly wrapper around :func:`normalize_text`."""
    return [normalize_text(text) for text in texts]
