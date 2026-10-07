"""Integration tests against a real Postgres + pgvector.

Run with: RAG_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag pytest -m integration
"""

import os
import uuid

import pytest

from production_rag.embeddings import HashingEmbedder
from production_rag.models import Chunk, Document

DATABASE_URL = os.getenv("RAG_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="RAG_TEST_DATABASE_URL not set"),
]


@pytest.fixture
def pg_store():  # type: ignore[no-untyped-def]
    from production_rag.store.pgvector import PgVectorStore

    store = PgVectorStore(DATABASE_URL or "", embedding_dim=64)
    with store._pool.connection() as conn:
        conn.execute("TRUNCATE documents CASCADE")
    yield store
    store.close()


def _ingest(store, doc_id: str, texts: list[str], **meta: str) -> None:  # type: ignore[no-untyped-def]
    embedder = HashingEmbedder(dim=64)
    chunks = [
        Chunk(id=f"{doc_id}:{i}", document_id=doc_id, index=i, text=t, metadata=dict(meta))
        for i, t in enumerate(texts)
    ]
    store.upsert(
        Document(id=doc_id, source=f"{doc_id}.md", title=doc_id), chunks, embedder.embed(texts)
    )


def test_vector_and_keyword_search(pg_store) -> None:  # type: ignore[no-untyped-def]
    doc = uuid.uuid4().hex[:8]
    _ingest(
        pg_store,
        doc,
        ["Refunds are processed within five business days", "Office hours are 9 to 5"],
    )
    [query] = HashingEmbedder(dim=64).embed(["refunds processed business days"])
    assert pg_store.vector_search(query, k=1)[0].chunk.id == f"{doc}:0"
    assert pg_store.keyword_search("how are refunds processed?", k=1)[0].chunk.id == f"{doc}:0"


def test_metadata_filters(pg_store) -> None:  # type: ignore[no-untyped-def]
    _ingest(pg_store, "hr", ["vacation policy"], department="hr")
    _ingest(pg_store, "it", ["vacation laptop policy"], department="it")
    results = pg_store.keyword_search("vacation policy", k=5, filters={"department": "it"})
    assert [r.chunk.document_id for r in results] == ["it"]


def test_upsert_replaces_and_delete_cascades(pg_store) -> None:  # type: ignore[no-untyped-def]
    _ingest(pg_store, "d1", ["old one", "old two"])
    _ingest(pg_store, "d1", ["new text"])
    assert [r.chunk.text for r in pg_store.keyword_search("old new text", k=10)] == ["new text"]
    assert pg_store.delete_document("d1") is True
    assert pg_store.keyword_search("new text", k=10) == []
    assert pg_store.list_documents() == []
