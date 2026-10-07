"""In-memory store: cosine similarity + Okapi BM25. Good for tests, evals and demos."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from threading import RLock

from production_rag.models import Chunk, Document, ScoredChunk
from production_rag.store.base import Filters, matches_filters
from production_rag.text import tokenize


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


class InMemoryStore:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._tokens: dict[str, list[str]] = {}
        self._lock = RLock()

    def upsert(
        self, document: Document, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        with self._lock:
            self.delete_document(document.id)
            self._documents[document.id] = document
            for chunk, vector in zip(chunks, embeddings, strict=True):
                self._chunks[chunk.id] = chunk
                self._vectors[chunk.id] = list(vector)
                self._tokens[chunk.id] = tokenize(chunk.text)

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            existed = self._documents.pop(document_id, None) is not None
            for chunk_id in [c.id for c in self._chunks.values() if c.document_id == document_id]:
                del self._chunks[chunk_id]
                del self._vectors[chunk_id]
                del self._tokens[chunk_id]
            return existed

    def list_documents(self) -> list[Document]:
        with self._lock:
            return list(self._documents.values())

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            return self._chunks.get(chunk_id)

    def _candidates(self, filters: Filters | None) -> list[Chunk]:
        return [c for c in self._chunks.values() if matches_filters(c.metadata, filters)]

    def vector_search(
        self, embedding: Sequence[float], k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        with self._lock:
            query_norm = _norm(embedding) or 1.0
            results = []
            for chunk in self._candidates(filters):
                vector = self._vectors[chunk.id]
                denom = (_norm(vector) or 1.0) * query_norm
                results.append(ScoredChunk(chunk=chunk, score=_dot(embedding, vector) / denom))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def keyword_search(
        self, query: str, k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        with self._lock:
            candidates = self._candidates(filters)
            if not candidates:
                return []
            # Corpus statistics are computed over the whole collection, not the filtered set,
            # so scores stay comparable regardless of filters.
            n_docs = len(self._tokens)
            avg_len = sum(len(t) for t in self._tokens.values()) / n_docs
            doc_freq: Counter[str] = Counter()
            for tokens in self._tokens.values():
                doc_freq.update(set(tokens) & query_terms)

            results = []
            for chunk in candidates:
                tokens = self._tokens[chunk.id]
                tf = Counter(tokens)
                score = 0.0
                for term in query_terms:
                    if term not in tf:
                        continue
                    idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                    norm = 1 - self.b + self.b * len(tokens) / avg_len
                    score += idf * tf[term] * (self.k1 + 1) / (tf[term] + self.k1 * norm)
                if score > 0:
                    results.append(ScoredChunk(chunk=chunk, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
