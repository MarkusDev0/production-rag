from itertools import pairwise

import pytest

from production_rag.ingestion.chunking import chunk_document, document_id_for, split_text
from production_rag.ingestion.loaders import LoadedDocument, LoadedPage

PARAGRAPHS = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) for i in range(10)
)


def test_chunks_respect_size_limit() -> None:
    chunks = split_text(PARAGRAPHS, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_no_content_is_lost() -> None:
    chunks = split_text(PARAGRAPHS, chunk_size=300, chunk_overlap=0)
    for i in range(10):
        for j in range(40):
            assert any(f"word{i}_{j}" in c for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_text(text, chunk_size=200, chunk_overlap=40)
    for prev, nxt in pairwise(chunks):
        assert prev.split()[-1] in nxt.split()


def test_prefers_paragraph_boundaries() -> None:
    text = "First paragraph is short.\n\nSecond paragraph is also short."
    assert split_text(text, chunk_size=40, chunk_overlap=0) == [
        "First paragraph is short.",
        "Second paragraph is also short.",
    ]


def test_hard_splits_unbroken_text() -> None:
    chunks = split_text("x" * 1000, chunk_size=300, chunk_overlap=0)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks) == "x" * 1000


def test_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, chunk_overlap=100)


def test_chunk_document_tracks_pages_and_ids() -> None:
    doc = LoadedDocument(
        source="handbook.pdf",
        title="Handbook",
        pages=[LoadedPage("Page one text.", page=1), LoadedPage("Page two text.", page=2)],
    )
    chunks = chunk_document(doc, extra_metadata={"team": "hr"})
    doc_id = document_id_for("handbook.pdf")
    assert [c.id for c in chunks] == [f"{doc_id}:0", f"{doc_id}:1"]
    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert all(c.metadata["team"] == "hr" for c in chunks)


def test_document_id_is_stable() -> None:
    assert document_id_for("a.pdf") == document_id_for("a.pdf") != document_id_for("b.pdf")
