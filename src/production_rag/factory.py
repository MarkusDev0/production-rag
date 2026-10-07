"""Builds a fully wired pipeline from settings. The only place that knows about
concrete providers; everything else depends on interfaces."""

from __future__ import annotations

from production_rag.config import Settings
from production_rag.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from production_rag.generation.answer import AnswerGenerator
from production_rag.generation.llm import LLM, AnthropicLLM, OpenAILLM
from production_rag.models import CostFn
from production_rag.pipeline import RAGPipeline
from production_rag.retrieval.hybrid import Retriever
from production_rag.retrieval.rerank import LLMReranker, NoopReranker, Reranker
from production_rag.store.base import VectorStore
from production_rag.store.memory import InMemoryStore


def _secret(value: object) -> str | None:
    getter = getattr(value, "get_secret_value", None)
    return getter() if callable(getter) else None


def build_cost_fn(settings: Settings) -> CostFn:
    pin, pout = settings.price_input_per_mtok, settings.price_output_per_mtok

    def cost(input_tokens: int, output_tokens: int) -> float | None:
        if pin is None or pout is None:
            return None
        return round((input_tokens * pin + output_tokens * pout) / 1_000_000, 6)

    return cost


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(
            settings.llm_model,
            api_key=_secret(settings.anthropic_api_key),
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    return OpenAILLM(
        settings.llm_model,
        api_key=_secret(settings.openai_api_key),
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(dim=settings.embedding_dim)
    return OpenAIEmbedder(
        settings.embedding_model,
        settings.embedding_dim,
        api_key=_secret(settings.openai_api_key),
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "memory":
        return InMemoryStore()
    from production_rag.store.pgvector import PgVectorStore

    return PgVectorStore(settings.database_url, settings.embedding_dim)


def build_pipeline(settings: Settings) -> RAGPipeline:
    llm = build_llm(settings)
    cost_fn = build_cost_fn(settings)
    embedder = build_embedder(settings)
    store = build_store(settings)
    reranker: Reranker = (
        LLMReranker(llm, cost_fn=cost_fn) if settings.reranker == "llm" else NoopReranker()
    )
    retriever = Retriever(
        store,
        embedder,
        mode=settings.retrieval_mode,
        reranker=reranker,
        candidate_k=settings.candidate_k,
    )
    return RAGPipeline(
        store=store,
        embedder=embedder,
        retriever=retriever,
        generator=AnswerGenerator(llm, cost_fn=cost_fn),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        default_top_k=settings.top_k,
    )
