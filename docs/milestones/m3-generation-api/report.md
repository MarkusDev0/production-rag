# M3 — Generation & API: report

**Status:** delivered  
**Release:** v0.3.0

## Delivered
| Area | Files |
|---|---|
| Answer generation | `generation/answer.py` |
| Orchestration | `pipeline.py`, `factory.py` |
| HTTP API | `api/main.py`, `api/schemas.py` |
| Containers | `Dockerfile`, `docker-compose.yml`, `.dockerignore` |

## Acceptance criteria
- [x] Citations always point to chunks that were actually given to the model
- [x] Unanswerable questions return `abstained: true`
- [x] Empty documents are rejected with a message pointing to OCR
- [x] Error mapping: 413 (too large), 415 (type), 422 (empty/invalid), 502 (fetch failed), 404 (unknown document)
- [x] `docker compose up` starts the API and pgvector with health checks
- [x] API tests run without network access

## Verification
15 new tests (53 total): citation ordering and validation, abstention, no-LLM-call on empty retrieval, prompt format, usage and cost; end-to-end pipeline including re-ingestion; API upload of a real PDF fixture through to a cited answer with page number, validation and error codes. The server was also started against a real Postgres to confirm migrations, ingestion and JSON logging.

## Decisions and deviations from the plan
- **Abstention uses a sentinel token** (`INSUFFICIENT_CONTEXT`) that is converted to a fixed user-facing message, so behavior doesn't depend on parsing free-form refusals.
- **Cost is opt-in**: prices are configured per deployment (`RAG_PRICE_*_PER_MTOK`) rather than hard-coded, since provider prices change.
- **The container runs as a non-root user** with a health check.
- Uploaded metadata must be a JSON object of strings, which keeps filter semantics simple and indexable.

## Known limitations
- No authentication; intended to run behind a gateway.
- Ingestion is synchronous; large batches should move to a queue.

## Next steps
- M4: evaluation harness, CI, documentation.
