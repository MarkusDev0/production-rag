"""Recursive, structure-aware chunking.

Text is split on the coarsest boundary that keeps pieces under the target size
(paragraphs, then lines, then sentences, then words), and pieces are merged back
greedily with a configurable overlap. Page numbers are preserved per chunk so
citations can point to the exact page.
"""

from __future__ import annotations

import hashlib

from production_rag.ingestion.loaders import LoadedDocument
from production_rag.models import Chunk

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def document_id_for(source: str) -> str:
    """Stable id per source, so re-ingesting a source replaces it instead of duplicating it."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _split_keep(text: str, sep: str) -> list[str]:
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p]


def _atomize(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _atomize(text, size, rest)
    pieces: list[str] = []
    for part in _split_keep(text, sep):
        pieces.extend(_atomize(part, size, rest) if len(part) > size else [part])
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) > size:
            chunks.append("".join(current))
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev)
            if carry_len + len(piece) > size:
                carry, carry_len = [], 0
            current, length = carry, carry_len
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current))
    return [c.strip() for c in chunks if c.strip()]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text.strip():
        return []
    return _merge(_atomize(text, chunk_size, SEPARATORS), chunk_size, chunk_overlap)


def chunk_document(
    doc: LoadedDocument,
    *,
    document_id: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    extra_metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    doc_id = document_id or document_id_for(doc.source)
    chunks: list[Chunk] = []
    for page in doc.pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            index = len(chunks)
            metadata: dict[str, object] = {"source": doc.source, "title": doc.title}
            if page.page is not None:
                metadata["page"] = page.page
            if extra_metadata:
                metadata.update(extra_metadata)
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{index}",
                    document_id=doc_id,
                    index=index,
                    text=piece,
                    metadata=metadata,
                )
            )
    return chunks
