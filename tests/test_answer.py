from production_rag.generation.answer import ABSTAIN_MESSAGE, ABSTAIN_TOKEN, AnswerGenerator
from production_rag.models import Chunk, ScoredChunk
from tests.conftest import FakeLLM


def _chunks() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                id=f"d:{i}",
                document_id="d",
                index=i,
                text=f"Fact number {i}.",
                metadata={"source": "handbook.pdf", "title": "Handbook", "page": i + 1},
            ),
            score=1.0,
        )
        for i in range(3)
    ]


def test_citations_map_to_sources_in_order_of_use() -> None:
    answer = AnswerGenerator(FakeLLM("A is true [2]. B is true [1][2].")).generate("q", _chunks())
    assert not answer.abstained
    assert [c.ref for c in answer.citations] == [2, 1]
    assert answer.citations[0].chunk_id == "d:1"
    assert answer.citations[0].page == 2


def test_invalid_citations_are_removed() -> None:
    answer = AnswerGenerator(FakeLLM("Claim [1]. Made up [7].")).generate("q", _chunks())
    assert "[7]" not in answer.answer
    assert [c.ref for c in answer.citations] == [1]


def test_abstains_when_model_signals_missing_context() -> None:
    answer = AnswerGenerator(FakeLLM(ABSTAIN_TOKEN)).generate("q", _chunks())
    assert answer.abstained
    assert answer.answer == ABSTAIN_MESSAGE
    assert answer.citations == []


def test_abstains_without_llm_call_when_nothing_retrieved() -> None:
    llm = FakeLLM()
    answer = AnswerGenerator(llm).generate("q", [])
    assert answer.abstained
    assert llm.calls == []


def test_prompt_contains_numbered_sources_with_pages() -> None:
    llm = FakeLLM()
    AnswerGenerator(llm).generate("What is fact 1?", _chunks())
    _, prompt = llm.calls[0]
    assert "[1] (Handbook, page 1)" in prompt
    assert "Question: What is fact 1?" in prompt


def test_usage_and_cost_are_reported() -> None:
    generator = AnswerGenerator(FakeLLM("Yes [1]."), cost_fn=lambda i, o: i * 0.001 + o * 0.002)
    usage = generator.generate("q", _chunks()).usage
    assert (usage.input_tokens, usage.output_tokens) == (100, 20)
    assert usage.cost_usd == 0.14
