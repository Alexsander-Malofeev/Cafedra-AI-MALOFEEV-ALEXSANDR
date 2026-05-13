from __future__ import annotations

import re

_SPACES_RE = re.compile(r"\s+")
_NOT_TEXT_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")


def clean_text(text: str) -> str:
    """Basic text cleanup for Russian support messages.

    The function is intentionally simple: lowercase, replace ё with е,
    remove extra punctuation and collapse spaces.
    """
    if text is None:
        return ""

    text = str(text).lower().replace("ё", "е")
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _NOT_TEXT_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def count_words(text: str) -> int:
    cleaned = clean_text(text)
    if not cleaned:
        return 0
    return len(cleaned.split())


def is_too_short(text: str, min_words: int = 2, min_chars: int = 8) -> bool:
    """Return True if user message is too short for a confident decision."""
    cleaned = clean_text(text)
    if not cleaned:
        return True
    return len(cleaned) < min_chars or len(cleaned.split()) < min_words
