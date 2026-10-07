"""Rerankers re-score retrieval candidates against the query.

The LLM reranker asks a model for a 0-10 relevance score per passage in a single
call. If the call or the JSON parsing fails, it falls back to the original order:
reranking should improve answers, never take the service down.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Protocol

from production_rag.generation.llm import LLM
from production_rag.models import CostFn, ScoredChunk, Usage

logger = logging.getLogger(__name__)

RERANK_SYSTEM = (
    "You are a search relevance judge. Score how well each passage helps answer the query, "
    "from 0 (irrelevant) to 10 (directly answers it). Judge only relevance, not writing quality."
)


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], Usage]: ...


class NoopReranker:
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        return list(candidates[:top_k]), Usage()


class LLMReranker:
    def __init__(
        self,
        llm: LLM,
        *,
        max_passage_chars: int = 1200,
        min_score: float = 0.0,
        cost_fn: CostFn | None = None,
    ) -> None:
        self.llm = llm
        self.max_passage_chars = max_passage_chars
        self.min_score = min_score
        self.cost_fn = cost_fn

    def _build_prompt(self, query: str, candidates: Sequence[ScoredChunk]) -> str:
        passages = "\n\n".join(
            f"[{i}] {c.chunk.text[: self.max_passage_chars]}" for i, c in enumerate(candidates)
        )
        return (
            f"Query: {query}\n\nPassages:\n{passages}\n\n"
            'Return JSON: {"scores": [{"id": <passage number>, "score": <0-10>}, ...]} '
            "with one entry per passage."
        )

    @staticmethod
    def _parse_scores(text: str, n: int) -> dict[int, float]:
        data = json.loads(text)
        scores: dict[int, float] = {}
        for item in data["scores"]:
            idx = int(item["id"])
            if 0 <= idx < n:
                scores[idx] = max(0.0, min(10.0, float(item["score"])))
        if not scores:
            raise ValueError("no valid scores in reranker output")
        return scores

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> tuple[list[ScoredChunk], Usage]:
        if len(candidates) <= 1:
            return list(candidates[:top_k]), Usage()
        try:
            response = self.llm.complete(
                RERANK_SYSTEM, self._build_prompt(query, candidates), json_mode=True, max_tokens=800
            )
        except Exception:
            logger.exception("reranker call failed; falling back to retrieval order")
            return list(candidates[:top_k]), Usage()

        usage = Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=self.cost_fn(response.input_tokens, response.output_tokens)
            if self.cost_fn
            else None,
        )
        try:
            scores = self._parse_scores(response.text, len(candidates))
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "unparseable reranker output; using retrieval order", extra={"error": str(exc)}
            )
            return list(candidates[:top_k]), usage

        # Unscored passages keep their place behind scored ones (stable sort on original rank).
        ranked = sorted(range(len(candidates)), key=lambda i: (-scores.get(i, -1.0), i))
        results = [
            ScoredChunk(chunk=candidates[i].chunk, score=scores.get(i, 0.0) / 10)
            for i in ranked
            if scores.get(i, 0.0) >= self.min_score
        ]
        return results[:top_k], usage
