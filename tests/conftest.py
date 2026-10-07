from __future__ import annotations

import pytest

from production_rag.embeddings import HashingEmbedder
from production_rag.ingestion.chunking import chunk_document, document_id_for
from production_rag.ingestion.loaders import load_text
from production_rag.models import Document
from production_rag.store.memory import InMemoryStore

CORPUS = {
    "refunds.md": (
        "Refund policy\n\nCustomers can request a full refund within 30 days of purchase. "
        "Refunds are issued to the original payment method within 5 business days."
    ),
    "security.md": (
        "Security policy\n\nAll employees must enable two-factor authentication. "
        "Passwords are rotated every 90 days and stored in the company vault."
    ),
    "shipping.md": (
        "Shipping\n\nStandard shipping takes 3 to 5 business days. "
        "Express shipping is delivered the next business day for orders placed before 2 pm."
    ),
}


def department_for(name: str) -> str:
    return "finance" if name == "refunds.md" else "ops"


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def seeded_store(store: InMemoryStore, embedder: HashingEmbedder) -> InMemoryStore:
    for name, text in CORPUS.items():
        loaded = load_text(text, source=name)
        doc = Document(id=document_id_for(name), source=name, title=loaded.title)
        chunks = chunk_document(
            loaded,
            chunk_size=300,
            chunk_overlap=40,
            extra_metadata={"department": department_for(name)},
        )
        store.upsert(doc, chunks, embedder.embed([c.text for c in chunks]))
    return store


# --- LLM test doubles -------------------------------------------------------------------------

import json  # noqa: E402
from collections.abc import Callable  # noqa: E402

from production_rag.generation.llm import LLMResponse  # noqa: E402


class FakeLLM:
    """Records calls and answers with a scripted responder."""

    def __init__(self, responder: Callable[[str, str], str] | str = "Answer [1].") -> None:
        self.model = "fake"
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 1024, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append((system, prompt))
        text = self.responder if isinstance(self.responder, str) else self.responder(system, prompt)
        return LLMResponse(text=text, input_tokens=100, output_tokens=20, model=self.model)


class FailingLLM:
    model = "failing"

    def complete(self, *args: object, **kwargs: object) -> LLMResponse:
        raise RuntimeError("provider unavailable")


def json_scores(scores: dict[int, float]) -> str:
    return json.dumps({"scores": [{"id": k, "score": v} for k, v in scores.items()]})


# --- Pipeline -----------------------------------------------------------------------------------

from production_rag.generation.answer import AnswerGenerator  # noqa: E402
from production_rag.pipeline import RAGPipeline  # noqa: E402
from production_rag.retrieval.hybrid import Retriever  # noqa: E402


@pytest.fixture
def make_pipeline(store: InMemoryStore, embedder: HashingEmbedder) -> Callable[..., RAGPipeline]:
    def factory(llm: FakeLLM | None = None, *, ingest: bool = True) -> RAGPipeline:
        pipeline = RAGPipeline(
            store=store,
            embedder=embedder,
            retriever=Retriever(store, embedder, mode="hybrid", candidate_k=10),
            generator=AnswerGenerator(llm or FakeLLM()),
            chunk_size=300,
            chunk_overlap=40,
            default_top_k=3,
        )
        if ingest:
            for name, text in CORPUS.items():
                pipeline.ingest(load_text(text, source=name), {"department": department_for(name)})
        return pipeline

    return factory
