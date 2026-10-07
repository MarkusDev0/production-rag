# M2 — Retrieval: plan

**Status:** planned  
**Target release:** v0.2.0

## Goal
Retrieve the right chunks reliably, for both natural-language questions and exact terms, from a persistent store.

## Scope
1. PostgreSQL + pgvector store implementing the M1 storage interface:
   - HNSW index for cosine similarity
   - generated `tsvector` column with GIN index for full-text search
   - JSONB metadata with GIN index for filters
   - transactional upsert (replace previous version of a document)
2. Provider-agnostic LLM client (OpenAI, Anthropic) with timeouts, retries and token usage reporting.
3. Hybrid retrieval combining vector and keyword candidates with Reciprocal Rank Fusion.
4. Reranking interface with an LLM-based implementation.

## Deliverables
- `store/pgvector.py`, `generation/llm.py`, `retrieval/hybrid.py`, `retrieval/rerank.py`
- Integration tests against a real Postgres + pgvector, skipped when no database is configured

## Acceptance criteria
- [ ] pgvector store passes the same behavioral tests as the in-memory store (search, filters, replace, delete)
- [ ] Retriever supports `vector`, `keyword` and `hybrid` modes behind one setting
- [ ] Fusion does not depend on raw score scales
- [ ] Reranker failures (provider error, malformed output) never fail the request
- [ ] Reranker token usage is reported for cost tracking

## Out of scope
- Cross-encoder rerankers (possible later through the same interface)
- Query rewriting / multi-query expansion

## Risks
| Risk | Mitigation |
|---|---|
| Postgres full-text AND semantics return nothing for long questions | Build an OR query from the stemmed terms |
| LLM reranker adds latency and cost | Make it optional (`RAG_RERANKER=none`); single batched call per query |
| Malformed JSON from the reranker | JSON mode + defensive parsing + fallback to retrieval order |
