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

LEET_MAP = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"}
)
LEET_CHARS = set("01345678@$")
PLACEHOLDER_RE = re.compile(r"^__\w+__$")


def normalize_text(text: str) -> str:
    """Lower-case a message and replace volatile entities with placeholder tokens."""
    text = str(text)
    shouty = " __allcaps__" if _shouty(text) else ""
    text = URL_RE.sub(" __url__ ", text)
    text = EMAIL_RE.sub(" __email__ ", text)
    text = MONEY_RE.sub(" __money__ ", text)
    text = PHONE_RE.sub(" __phone__ ", text)
    text = text.lower()
    deleeted = _deleet(text)
    text = NUMBER_RE.sub(" __number__ ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return " ".join(part for part in (text, deleeted, shouty.strip()) if part)


def _deleet(text: str) -> str:
    """Recover words hidden behind digit substitutions, e.g. ``fr33 m0ney`` -> ``free money``.

    Only tokens that mix letters with leet characters are rewritten, and the result is
    appended to the message rather than replacing it, so plain text is left untouched.
    """
    recovered = [
        token.translate(LEET_MAP)
        for token in text.split()
        if any(char.isalpha() for char in token)
        and LEET_CHARS.intersection(token)
        and not PLACEHOLDER_RE.match(token)
    ]
    return " ".join(recovered)


def _shouty(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 10:
        return False
    upper = sum(char.isupper() for char in letters)
    return upper / len(letters) > 0.5


def normalize_corpus(texts: Iterable[str]) -> list[str]:
    """Vectoriser-friendly wrapper around :func:`normalize_text`."""
    return [normalize_text(text) for text in texts]
