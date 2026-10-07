import pytest

from production_rag.embeddings import HashingEmbedder
from production_rag.models import Chunk, ScoredChunk
from production_rag.retrieval.hybrid import Retriever, reciprocal_rank_fusion
from production_rag.store.memory import InMemoryStore


def _sc(chunk_id: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, document_id="d", index=0, text=chunk_id), score=score
    )


def test_rrf_rewards_agreement_between_lists() -> None:
    vector = [_sc("a"), _sc("b"), _sc("c")]
    keyword = [_sc("d"), _sc("b"), _sc("e")]
    fused = [r.chunk.id for r in reciprocal_rank_fusion([vector, keyword])]
    assert fused[0] == "b"  # ranked 2nd in both lists beats 1st in one
    assert set(fused) == {"a", "b", "c", "d", "e"}


def test_rrf_ignores_raw_score_scales() -> None:
    a = [_sc("x", 0.9), _sc("y", 0.1)]
    b = [_sc("x", 900.0), _sc("y", 1.0)]
    assert [r.chunk.id for r in reciprocal_rank_fusion([a, b])] == ["x", "y"]


def test_hybrid_retriever_finds_exact_terms(
    seeded_store: InMemoryStore, embedder: HashingEmbedder
) -> None:
    retriever = Retriever(seeded_store, embedder, mode="hybrid")
    results, _ = retriever.retrieve("How long does express shipping take?", top_k=1)
    assert "Express shipping" in results[0].chunk.text


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_retriever_modes(seeded_store: InMemoryStore, embedder: HashingEmbedder, mode: str) -> None:
    retriever = Retriever(seeded_store, embedder, mode=mode)  # type: ignore[arg-type]
    results, _ = retriever.retrieve("two-factor authentication", top_k=1)
    assert "two-factor" in results[0].chunk.text


def test_retriever_applies_filters(seeded_store: InMemoryStore, embedder: HashingEmbedder) -> None:
    retriever = Retriever(seeded_store, embedder)
    results, _ = retriever.retrieve("policy days", top_k=5, filters={"department": "finance"})
    assert results and all(r.chunk.metadata["department"] == "finance" for r in results)
