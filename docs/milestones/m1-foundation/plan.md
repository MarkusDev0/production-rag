# M1 — Foundation & ingestion: plan

**Status:** planned  
**Target release:** v0.1.0

## Goal
Set up a maintainable project skeleton and turn raw documents into clean, traceable chunks that later milestones can index and retrieve.

## Scope
1. Project tooling: packaging (`pyproject.toml`), ruff, mypy in strict mode, pytest, pre-commit, Makefile.
2. Typed settings loaded from environment variables, validated at startup.
3. Domain models shared by all layers: `Document`, `Chunk`, `ScoredChunk`.
4. Structured JSON logging with request-id support.
5. Loaders for PDF (per page), HTML (boilerplate removed), Markdown/plain text, and URLs.
6. Chunking that respects document structure, with configurable size and overlap, preserving page numbers.
7. Embedding interface with an OpenAI backend and an offline backend for tests.
8. A storage interface and an in-memory implementation with vector and keyword (BM25) search and metadata filters.

## Deliverables
- `src/production_rag/` package with the modules above
- Unit tests for loaders, chunking, embeddings and the in-memory store

## Acceptance criteria
- [ ] PDF pages keep their page number through chunking
- [ ] HTML scripts, navigation and footers are excluded from extracted text
- [ ] No chunk exceeds the configured size; no text is lost when overlap is 0
- [ ] Re-ingesting the same source replaces it instead of duplicating it
- [ ] Keyword search ranks exact-term matches first; metadata filters restrict results
- [ ] Tests run offline with no API key
- [ ] `ruff` and `mypy --strict` pass

## Out of scope
- Persistent storage (M2), answer generation and HTTP API (M3), evaluation (M4)
- OCR for scanned PDFs

## Risks
| Risk | Mitigation |
|---|---|
| PDF text extraction quality varies by producer | Keep loaders isolated so a different extractor can be swapped in |
| Chunk boundaries split facts across chunks | Overlap between consecutive chunks; tune in M4 using the eval set |
