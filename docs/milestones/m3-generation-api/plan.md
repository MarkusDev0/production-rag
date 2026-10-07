# M3 — Generation & API: plan

**Status:** planned  
**Target release:** v0.3.0

## Goal
Turn retrieved chunks into answers users can trust and expose the system over a clean HTTP API that runs with one command.

## Scope
1. Answer generation from numbered sources with `[n]` citations.
2. Post-generation citation validation: references to sources that weren't provided are removed.
3. Abstention: an explicit flag when the sources don't contain the answer; no LLM call when nothing is retrieved.
4. Token and cost tracking per request, summed across reranking and generation.
5. Pipeline orchestrating ingestion and querying, with per-stage logging.
6. FastAPI service: upload, URL ingestion, list, delete, query, health; request IDs and response timing headers.
7. Dockerfile and Compose stack (API + pgvector).

## Deliverables
- `generation/answer.py`, `pipeline.py`, `factory.py`, `api/`
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`

## Acceptance criteria
- [ ] Citations in the response always point to chunks that were actually given to the model
- [ ] Unanswerable questions return `abstained: true` instead of a guess
- [ ] Empty documents (e.g. scanned PDFs) are rejected with a clear message
- [ ] API maps errors to correct status codes (413, 415, 422, 502, 404)
- [ ] `docker compose up` starts a working stack
- [ ] API covered by tests that run without network access

## Out of scope
- Authentication and multi-tenancy
- Streaming responses
- Background ingestion queue

## Risks
| Risk | Mitigation |
|---|---|
| Model ignores the citation format | Strict system prompt; validation flags answers with no valid citations in logs |
| Model answers from prior knowledge | Prompt restricts to sources; eval set in M4 includes unanswerable questions |
| Large uploads exhaust memory | Configurable size limit enforced while reading |
