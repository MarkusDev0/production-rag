import pytest

from production_rag.ingestion.loaders import LoadedDocument, LoadedPage, load_text
from production_rag.pipeline import EmptyDocumentError
from tests.conftest import FakeLLM


def test_query_end_to_end(make_pipeline) -> None:  # type: ignore[no-untyped-def]
    llm = FakeLLM("Refunds can be requested within 30 days [1].")
    answer = make_pipeline(llm).query("What is the refund window?")
    assert not answer.abstained
    assert answer.citations[0].source == "refunds.md"
    _, prompt = llm.calls[0]
    assert "30 days" in prompt  # the right context reached the model


def test_reingesting_same_source_replaces_document(make_pipeline) -> None:  # type: ignore[no-untyped-def]
    pipeline = make_pipeline(ingest=False)
    pipeline.ingest(load_text("Version one.", source="faq.md"))
    pipeline.ingest(load_text("Version two.", source="faq.md"))
    assert len(pipeline.store.list_documents()) == 1
    assert pipeline.retrieve("version", top_k=5)[0].chunk.text == "Version two."


def test_empty_documents_are_rejected(make_pipeline) -> None:  # type: ignore[no-untyped-def]
    empty = LoadedDocument(source="scan.pdf", title="scan", pages=[LoadedPage("", page=1)])
    with pytest.raises(EmptyDocumentError):
        make_pipeline(ingest=False).ingest(empty)
