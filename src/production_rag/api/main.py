"""FastAPI application.

Run locally:  uvicorn production_rag.api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status

from production_rag import __version__
from production_rag.api.schemas import (
    HealthResponse,
    IngestResponse,
    IngestUrlRequest,
    QueryRequest,
    QueryResponse,
)
from production_rag.config import get_settings
from production_rag.ingestion.loaders import UnsupportedFileType, fetch_url, load_file
from production_rag.logging_config import configure_logging, request_id_var
from production_rag.models import Document
from production_rag.pipeline import EmptyDocumentError, RAGPipeline

logger = logging.getLogger(__name__)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level)
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            from production_rag.factory import build_pipeline

            app.state.pipeline = build_pipeline(settings)
        yield
        close = getattr(app.state.pipeline.store, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="production-rag",
        version=__version__,
        description="Hybrid-search RAG with reranking, grounded citations and evaluation.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    def get_pipeline(request: Request) -> RAGPipeline:
        return request.app.state.pipeline  # type: ignore[no-any-return]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
    def ingest_file(
        request: Request,
        file: Annotated[UploadFile, File()],
        metadata: Annotated[str | None, Form(description="JSON object of string metadata")] = None,
    ) -> IngestResponse:
        data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
        extra = _parse_metadata(metadata)
        try:
            loaded = load_file(file.filename or "upload", data)
            result = get_pipeline(request).ingest(loaded, extra)
        except UnsupportedFileType as exc:
            raise HTTPException(415, str(exc)) from exc
        except EmptyDocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        return IngestResponse(document=result.document, chunk_count=result.chunk_count)

    @app.post("/documents/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
    def ingest_url(request: Request, body: IngestUrlRequest) -> IngestResponse:
        try:
            loaded = fetch_url(str(body.url))
            result = get_pipeline(request).ingest(loaded, body.metadata)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Could not fetch URL: {exc}") from exc
        except UnsupportedFileType as exc:
            raise HTTPException(415, str(exc)) from exc
        except EmptyDocumentError as exc:
            raise HTTPException(422, str(exc)) from exc
        return IngestResponse(document=result.document, chunk_count=result.chunk_count)

    @app.get("/documents", response_model=list[Document])
    def list_documents(request: Request) -> list[Document]:
        return get_pipeline(request).store.list_documents()

    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(request: Request, document_id: str) -> Response:
        if not get_pipeline(request).store.delete_document(document_id):
            raise HTTPException(404, "Document not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/query", response_model=QueryResponse)
    def query(request: Request, body: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        answer = get_pipeline(request).query(body.question, body.top_k, body.filters)
        return QueryResponse(
            **answer.model_dump(), latency_ms=round((time.perf_counter() - started) * 1000, 1)
        )

    return app


def _parse_metadata(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "metadata must be a JSON object") from exc
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        raise HTTPException(422, "metadata must be a JSON object of string values")
    return {str(k): v for k, v in value.items()}


app = create_app()
