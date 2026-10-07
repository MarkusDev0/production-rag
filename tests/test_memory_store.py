from production_rag.embeddings import HashingEmbedder
from production_rag.models import Chunk, Document
from production_rag.store.memory import InMemoryStore


def _add(
    store: InMemoryStore, embedder: HashingEmbedder, doc_id: str, texts: list[str], **meta: str
) -> None:
    chunks = [
        Chunk(id=f"{doc_id}:{i}", document_id=doc_id, index=i, text=t, metadata=dict(meta))
        for i, t in enumerate(texts)
    ]
    store.upsert(Document(id=doc_id, source=doc_id, title=doc_id), chunks, embedder.embed(texts))


def test_keyword_search_ranks_matching_chunk_first(
    store: InMemoryStore, embedder: HashingEmbedder
) -> None:
    _add(store, embedder, "d1", ["refunds are issued within 30 days", "shipping takes five days"])
    results = store.keyword_search("refund 30 days", k=2)
    assert results[0].chunk.id == "d1:0"


def test_keyword_search_ignores_stopword_only_queries(
    store: InMemoryStore, embedder: HashingEmbedder
) -> None:
    _add(store, embedder, "d1", ["some text"])
    assert store.keyword_search("the and of", k=5) == []


def test_vector_search_returns_best_match(store: InMemoryStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "d1", ["two factor authentication required", "office opening hours"])
    [query] = embedder.embed(["authentication two factor"])
    assert store.vector_search(query, k=1)[0].chunk.id == "d1:0"


def test_filters_restrict_results(store: InMemoryStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "hr", ["vacation policy days"], department="hr")
    _add(store, embedder, "it", ["vacation laptop policy"], department="it")
    results = store.keyword_search("vacation policy", k=5, filters={"department": "it"})
    assert [r.chunk.document_id for r in results] == ["it"]


def test_upsert_replaces_previous_version(store: InMemoryStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "d1", ["old a", "old b", "old c"])
    _add(store, embedder, "d1", ["new text"])
    assert len(store.list_documents()) == 1
    assert [r.chunk.text for r in store.keyword_search("old new text", k=10)] == ["new text"]


def test_delete_document(store: InMemoryStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "d1", ["hello world"])
    assert store.delete_document("d1") is True
    assert store.delete_document("d1") is False
    assert store.keyword_search("hello", k=5) == []


def test_get_chunk(store: InMemoryStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "d1", ["hello world"])
    chunk = store.get_chunk("d1:0")
    assert chunk is not None and chunk.text == "hello world"
    assert store.get_chunk("missing") is None
