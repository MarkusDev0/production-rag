"""Request/response contracts for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl

from production_rag.models import Answer, Document


class IngestResponse(BaseModel):
    document: Document
    chunk_count: int


class IngestUrlRequest(BaseModel):
    url: HttpUrl
    metadata: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: dict[str, Any] | None = Field(
        default=None, description='Exact-match metadata filters, e.g. {"department": "hr"}'
    )


class QueryResponse(Answer):
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
