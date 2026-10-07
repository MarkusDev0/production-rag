# M1 — Foundation & ingestion: report

**Status:** delivered  
**Release:** v0.1.0

## Delivered
| Area | Files |
|---|---|
| Tooling | `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.env.example` |
| Settings & models | `config.py`, `models.py`, `logging_config.py`, `text.py` |
| Loaders | `ingestion/loaders.py` |
| Chunking | `ingestion/chunking.py` |
| Embeddings | `embeddings.py` (`OpenAIEmbedder`, `HashingEmbedder`) |
| Storage | `store/base.py`, `store/memory.py` |

## Acceptance criteria
- [x] PDF pages keep their page number through chunking
- [x] HTML scripts, navigation and footers are excluded from extracted text
- [x] No chunk exceeds the configured size; no text is lost when overlap is 0
- [x] Re-ingesting the same source replaces it instead of duplicating it
- [x] Keyword search ranks exact-term matches first; metadata filters restrict results
- [x] Tests run offline with no API key
- [x] `ruff` and `mypy --strict` pass

## Verification
21 unit tests across loaders, chunking, embeddings and the in-memory store, all passing. Network calls in loader tests use `httpx.MockTransport`.

## Decisions and deviations from the plan
- **Document ids are derived from the source** (`sha256(source)[:16]`), which makes re-ingestion idempotent without a lookup table.
- **Settings validate cross-field rules** (`chunk_overlap < chunk_size`, `candidate_k >= top_k`) so misconfiguration fails at startup, not mid-request.
- **BM25 statistics use the whole collection**, not the filtered subset, so scores stay comparable regardless of filters.
- Added an offline `HashingEmbedder` (not in the original plan) so tests and evals never need network access. It is lexical only and documented as not for production.

## Known limitations
- No OCR: scanned PDFs produce empty text (to be rejected explicitly in M3).
- In-memory BM25 has no stemming.

## Next steps
- M2: persistent pgvector store and hybrid retrieval.
