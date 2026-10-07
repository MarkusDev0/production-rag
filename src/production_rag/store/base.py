"""Storage interface. Every backend supports vector search, keyword search and
exact-match metadata filters, so retrieval logic stays backend-agnostic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from production_rag.models import Chunk, Document, ScoredChunk

Filters = dict[str, Any]


def matches_filters(metadata: dict[str, Any], filters: Filters | None) -> bool:
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


class VectorStore(Protocol):
    def upsert(
        self, document: Document, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Insert a document and its chunks, replacing any previous version."""

    def delete_document(self, document_id: str) -> bool: ...

    def list_documents(self) -> list[Document]: ...

    def vector_search(
        self, embedding: Sequence[float], k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...

    def keyword_search(
        self, query: str, k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]: ...
