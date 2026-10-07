from pathlib import Path

import httpx
import pytest

from production_rag.ingestion.loaders import (
    UnsupportedFileType,
    fetch_url,
    load_file,
    load_html,
    load_pdf,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_pdf_keeps_page_numbers() -> None:
    doc = load_pdf((FIXTURES / "sample.pdf").read_bytes(), source="sample.pdf")
    assert doc.title == "Sample Handbook"
    assert [p.page for p in doc.pages] == [1, 2]
    assert "24 months" in doc.pages[0].text
    assert "one business day" in doc.pages[1].text


def test_load_html_strips_boilerplate() -> None:
    html = """
    <html><head><title>Docs</title><script>var x = 1;</script></head>
    <body><nav>Home | About</nav><main><h1>Install</h1><p>Run   make install.</p></main>
    <footer>Copyright</footer></body></html>
    """
    doc = load_html(html, source="https://example.com/docs")
    text = doc.pages[0].text
    assert doc.title == "Docs"
    assert "Install" in text and "Run make install." in text
    assert "var x" not in text and "Home | About" not in text and "Copyright" not in text


def test_load_file_dispatches_by_extension() -> None:
    assert load_file("notes.md", b"# Title\n\nBody").pages[0].text == "# Title\n\nBody"
    with pytest.raises(UnsupportedFileType):
        load_file("archive.zip", b"PK")


def test_fetch_url_uses_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><title>T</title><body><p>Hello world</p></body></html>",
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        doc = fetch_url("https://example.com/page", client=client)
    assert doc.title == "T"
    assert doc.metadata["url"] == "https://example.com/page"
    assert "Hello world" in doc.pages[0].text
