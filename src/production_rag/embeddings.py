"""Embedding backends behind one small interface."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from production_rag.text import tokenize


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class HashingEmbedder:
    """Deterministic, dependency-free embedder (signed feature hashing of unigrams).

    It captures lexical overlap only, not meaning. It exists so the whole system,
    tests and the eval harness run offline without an API key. Use a real model
    in production.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return _l2_normalize(vector)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """OpenAI embeddings with batching. Retries/backoff are handled by the SDK."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        api_key: str | None = None,
        batch_size: int = 96,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            response = self._client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors
