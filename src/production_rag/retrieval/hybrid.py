"""Hybrid retrieval: vector + keyword candidates fused with Reciprocal Rank Fusion.

RRF only uses ranks, so it needs no score calibration between cosine similarity
and BM25 / ts_rank, which live on completely different scales.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Literal

from production_rag.embeddings import Embedder
from production_rag.models import ScoredChunk, Usage
from production_rag.retrieval.rerank import NoopReranker, Reranker
from production_rag.store.base import Filters, VectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "keyword", "hybrid"]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]], *, k: int = 60
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, ScoredChunk] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item.chunk.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(chunk_id, item)
    fused = [ScoredChunk(chunk=by_id[cid].chunk, score=score) for cid, score in scores.items()]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        mode: RetrievalMode = "hybrid",
        reranker: Reranker | None = None,
        candidate_k: int = 20,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.mode = mode
        self.reranker = reranker or NoopReranker()
        self.candidate_k = candidate_k

    def candidates(self, query: str, filters: Filters | None = None) -> list[ScoredChunk]:
        lists: list[list[ScoredChunk]] = []
        if self.mode in ("vector", "hybrid"):
            [vector] = self.embedder.embed([query])
            lists.append(self.store.vector_search(vector, self.candidate_k, filters))
        if self.mode in ("keyword", "hybrid"):
            lists.append(self.store.keyword_search(query, self.candidate_k, filters))
        if len(lists) == 1:
            return lists[0]
        return reciprocal_rank_fusion(lists)[: self.candidate_k]

    def retrieve(
        self, query: str, top_k: int, filters: Filters | None = None
    ) -> tuple[list[ScoredChunk], Usage]:
        started = time.perf_counter()
        candidates = self.candidates(query, filters)
        results, usage = self.reranker.rerank(query, candidates, top_k)
        logger.info(
            "retrieval complete",
            extra={
                "mode": self.mode,
                "reranker": type(self.reranker).__name__,
                "candidates": len(candidates),
                "returned": len(results),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return results, usage
