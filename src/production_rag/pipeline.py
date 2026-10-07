"""The RAG pipeline: ingestion on one side, question answering on the other."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from production_rag.embeddings import Embedder
from production_rag.generation.answer import AnswerGenerator
from production_rag.ingestion.chunking import chunk_document, document_id_for
from production_rag.ingestion.loaders import LoadedDocument
from production_rag.models import Answer, Document, ScoredChunk
from production_rag.retrieval.hybrid import Retriever
from production_rag.store.base import Filters, VectorStore

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class IngestResult:
    document: Document
    chunk_count: int


class RAGPipeline:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        retriever: Retriever,
        generator: AnswerGenerator,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        default_top_k: int = 5,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_top_k = default_top_k

    def ingest(
        self, loaded: LoadedDocument, metadata: dict[str, str] | None = None
    ) -> IngestResult:
        if loaded.is_empty:
            raise EmptyDocumentError(
                f"No extractable text in {loaded.source}. Scanned PDFs need OCR first."
            )
        started = time.perf_counter()
        document = Document(
            id=document_id_for(loaded.source),
            source=loaded.source,
            title=loaded.title,
            metadata={**loaded.metadata, **(metadata or {})},
        )
        chunks = chunk_document(
            loaded,
            document_id=document.id,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            extra_metadata=metadata,
        )
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.upsert(document, chunks, embeddings)
        logger.info(
            "document ingested",
            extra={
                "document_id": document.id,
                "source": document.source,
                "chunks": len(chunks),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return IngestResult(document=document, chunk_count=len(chunks))

    def retrieve(
        self, question: str, top_k: int | None = None, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        results, _ = self.retriever.retrieve(question, top_k or self.default_top_k, filters)
        return results

    def query(
        self, question: str, top_k: int | None = None, filters: Filters | None = None
    ) -> Answer:
        started = time.perf_counter()
        chunks, retrieval_usage = self.retriever.retrieve(
            question, top_k or self.default_top_k, filters
        )
        answer = self.generator.generate(question, chunks)
        answer.usage = retrieval_usage + answer.usage
        logger.info(
            "query answered",
            extra={
                "abstained": answer.abstained,
                "citations": len(answer.citations),
                "input_tokens": answer.usage.input_tokens,
                "output_tokens": answer.usage.output_tokens,
                "cost_usd": answer.usage.cost_usd,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return answer
