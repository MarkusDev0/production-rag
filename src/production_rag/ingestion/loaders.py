"""Turn raw inputs (PDF, HTML, plain text, URLs) into normalized page text."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from production_rag.text import normalize_whitespace

# Elements that are almost always navigation or boilerplate, not content.
_HTML_NOISE = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg"]


class UnsupportedFileType(ValueError):
    pass


@dataclass
class LoadedPage:
    text: str
    page: int | None = None  # 1-based for paginated sources, None otherwise


@dataclass
class LoadedDocument:
    source: str
    title: str
    pages: list[LoadedPage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(p.text.strip() for p in self.pages)


def load_pdf(data: bytes, source: str, title: str | None = None) -> LoadedDocument:
    reader = PdfReader(io.BytesIO(data))
    pages = [
        LoadedPage(text=normalize_whitespace(page.extract_text() or ""), page=i)
        for i, page in enumerate(reader.pages, start=1)
    ]
    meta_title = None
    if reader.metadata is not None and reader.metadata.title:
        meta_title = str(reader.metadata.title)
    return LoadedDocument(
        source=source,
        title=title or meta_title or PurePath(source).stem,
        pages=pages,
        metadata={"content_type": "application/pdf", "page_count": len(pages)},
    )


def load_html(html: str, source: str, title: str | None = None) -> LoadedDocument:
    soup = BeautifulSoup(html, "html.parser")
    page_title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_HTML_NOISE):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = normalize_whitespace(root.get_text("\n"))
    return LoadedDocument(
        source=source,
        title=title or page_title or source,
        pages=[LoadedPage(text=text)],
        metadata={"content_type": "text/html"},
    )


def load_text(text: str, source: str, title: str | None = None) -> LoadedDocument:
    return LoadedDocument(
        source=source,
        title=title or PurePath(source).stem,
        pages=[LoadedPage(text=normalize_whitespace(text))],
        metadata={"content_type": "text/plain"},
    )


def load_file(filename: str, data: bytes) -> LoadedDocument:
    suffix = PurePath(filename).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(data, source=filename)
    if suffix in {".html", ".htm"}:
        return load_html(data.decode("utf-8", errors="replace"), source=filename)
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text(data.decode("utf-8", errors="replace"), source=filename)
    raise UnsupportedFileType(f"Unsupported file type: {suffix or 'unknown'}")


def fetch_url(
    url: str, *, timeout: float = 20.0, client: httpx.Client | None = None
) -> LoadedDocument:
    """Download a URL and load it as PDF or HTML based on the response content type."""
    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "production-rag/0.4 (+https://github.com)"},
    )
    try:
        response = http.get(url)
        response.raise_for_status()
    finally:
        if owns_client:
            http.close()

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type == "application/pdf" or url.lower().endswith(".pdf"):
        doc = load_pdf(response.content, source=url)
    elif content_type in {"text/html", "application/xhtml+xml", ""}:
        doc = load_html(response.text, source=url)
    elif content_type.startswith("text/"):
        doc = load_text(response.text, source=url, title=url)
    else:
        raise UnsupportedFileType(f"Unsupported content type: {content_type}")
    doc.metadata["url"] = url
    return doc
