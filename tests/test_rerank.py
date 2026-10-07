from production_rag.models import Chunk, ScoredChunk
from production_rag.retrieval.rerank import LLMReranker, NoopReranker
from tests.conftest import FailingLLM, FakeLLM, json_scores


def _candidates(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id=f"c{i}", document_id="d", index=i, text=f"text {i}"), score=1.0)
        for i in range(n)
    ]


def test_llm_reranker_reorders_by_score() -> None:
    llm = FakeLLM(json_scores({0: 2, 1: 9, 2: 5}))
    results, usage = LLMReranker(llm).rerank("q", _candidates(3), top_k=2)
    assert [r.chunk.id for r in results] == ["c1", "c2"]
    assert results[0].score == 0.9
    assert usage.input_tokens == 100


def test_llm_reranker_drops_below_min_score() -> None:
    llm = FakeLLM(json_scores({0: 1, 1: 8}))
    results, _ = LLMReranker(llm, min_score=3).rerank("q", _candidates(2), top_k=5)
    assert [r.chunk.id for r in results] == ["c1"]


def test_llm_reranker_falls_back_on_bad_json() -> None:
    results, _ = LLMReranker(FakeLLM("not json")).rerank("q", _candidates(3), top_k=2)
    assert [r.chunk.id for r in results] == ["c0", "c1"]


def test_llm_reranker_falls_back_on_provider_error() -> None:
    results, usage = LLMReranker(FailingLLM()).rerank("q", _candidates(3), top_k=2)
    assert [r.chunk.id for r in results] == ["c0", "c1"]
    assert usage.input_tokens == 0


def test_llm_reranker_ignores_out_of_range_ids() -> None:
    llm = FakeLLM(json_scores({7: 10, 1: 6}))
    results, _ = LLMReranker(llm).rerank("q", _candidates(2), top_k=2)
    assert [r.chunk.id for r in results] == ["c1", "c0"]


def test_llm_reranker_tracks_cost() -> None:
    reranker = LLMReranker(FakeLLM(json_scores({0: 1, 1: 2})), cost_fn=lambda i, o: (i + o) / 1000)
    _, usage = reranker.rerank("q", _candidates(2), top_k=1)
    assert usage.cost_usd == 0.12


def test_noop_reranker_truncates() -> None:
    results, _ = NoopReranker().rerank("q", _candidates(5), top_k=3)
    assert len(results) == 3
