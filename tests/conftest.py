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
