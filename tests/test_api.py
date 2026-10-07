import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from production_rag.api.main import create_app
from tests.conftest import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(make_pipeline):  # type: ignore[no-untyped-def]
    pipeline = make_pipeline(FakeLLM("Warranty lasts 24 months [1]."), ingest=False)
    with TestClient(create_app(pipeline)) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "x-request-id" in response.headers


def test_upload_and_query_pdf(client: TestClient) -> None:
    with (FIXTURES / "sample.pdf").open("rb") as f:
        upload = client.post(
            "/documents",
            files={"file": ("sample.pdf", f, "application/pdf")},
            data={"metadata": json.dumps({"team": "support"})},
        )
    assert upload.status_code == 201
    body = upload.json()
    assert body["chunk_count"] == 2
    assert body["document"]["metadata"]["team"] == "support"

    answer = client.post("/query", json={"question": "How long is the warranty?"})
    assert answer.status_code == 200
    data = answer.json()
    assert data["abstained"] is False
    assert data["citations"][0]["page"] == 1
    assert data["latency_ms"] >= 0


def test_unsupported_file_type(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("a.zip", b"PK", "application/zip")})
    assert response.status_code == 415


def test_invalid_metadata(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("a.txt", b"hello", "text/plain")}, data={"metadata": "[1]"}
    )
    assert response.status_code == 422


def test_list_and_delete_documents(client: TestClient) -> None:
    client.post("/documents", files={"file": ("a.txt", b"hello world", "text/plain")})
    docs = client.get("/documents").json()
    assert len(docs) == 1
    assert client.delete(f"/documents/{docs[0]['id']}").status_code == 204
    assert client.delete(f"/documents/{docs[0]['id']}").status_code == 404


def test_query_validation(client: TestClient) -> None:
    assert client.post("/query", json={"question": "?"}).status_code == 422
