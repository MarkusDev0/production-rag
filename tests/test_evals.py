"""Smoke tests for the evaluation harness, so it can't silently break."""

import pytest
from evals import run_evals

from production_rag.config import Settings
from production_rag.embeddings import HashingEmbedder
from production_rag.generation.answer import ABSTAIN_TOKEN
from tests.conftest import FakeLLM


def test_retrieval_eval_runs_offline() -> None:
    settings = Settings(chunk_size=300, chunk_overlap=50, embedding_provider="hashing")
    result = run_evals.evaluate(settings, HashingEmbedder(512), "hybrid", "none", 3, False)
    assert 0.8 <= result.hit_at_k <= 1.0
    assert result.answer_accuracy is None


def test_generation_eval_scores_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    def respond(system: str, prompt: str) -> str:
        return "The allowance is 65 EUR per day [1]." if "meal" in prompt else ABSTAIN_TOKEN

    monkeypatch.setattr(run_evals, "build_llm", lambda settings: FakeLLM(respond))
    settings = Settings(chunk_size=300, chunk_overlap=50, embedding_provider="hashing")
    result = run_evals.evaluate(settings, HashingEmbedder(512), "hybrid", "none", 3, True)
    assert result.abstention_accuracy == 1.0
    assert result.answer_accuracy is not None and 0 < result.answer_accuracy < 0.2
    assert result.grounded_citation_rate is not None and result.grounded_citation_rate > 0


def test_cli_quality_gate(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(run_evals, "RESULTS_DIR", tmp_path)
    assert run_evals.main(["--modes", "hybrid", "--min-hit-rate", "0.5"]) == 0
    assert run_evals.main(["--modes", "hybrid", "--min-hit-rate", "1.01"]) == 1
