"""Small text utilities shared by chunking, keyword search and embeddings."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS_TEXT = """
a an and are as at be but by can do does for from has have how i if in into is it its may no
not of on or our should so than that the their them then there these they this to was we were
what when where which who will with you your
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and limit blank lines, keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
