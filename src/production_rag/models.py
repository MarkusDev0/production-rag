"""Domain models shared across ingestion, retrieval and generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# (input_tokens, output_tokens) -> USD, or None when pricing isn't configured.
CostFn = Callable[[int, int], float | None]


class Document(BaseModel):
    id: str
    source: str
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    document_id: str
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A source the answer actually referenced, as `[n]` in the text."""

    ref: int
    chunk_id: str
    document_id: str
    source: str
    title: str
    page: int | None = None
    snippet: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost: float | None
        if self.cost_usd is None and other.cost_usd is None:
            cost = None
        else:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class Answer(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    usage: Usage = Field(default_factory=Usage)
