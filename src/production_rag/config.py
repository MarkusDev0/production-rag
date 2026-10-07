"""Application settings, loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    # Providers
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3

    embedding_provider: Literal["openai", "hashing"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Storage
    vector_store: Literal["pgvector", "memory"] = "pgvector"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Ingestion
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    max_upload_mb: int = 20

    # Retrieval
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    reranker: Literal["llm", "none"] = "llm"
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=200)

    # Cost tracking (USD per 1M tokens). Left unset, cost is reported as null.
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
