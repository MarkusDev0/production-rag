# M2 — Retrieval: report

**Status:** delivered  
**Release:** v0.2.0

## Delivered
| Area | Files |
|---|---|
| Persistent store | `store/pgvector.py` |
| LLM client | `generation/llm.py` (`OpenAILLM`, `AnthropicLLM`) |
| Hybrid retrieval | `retrieval/hybrid.py` (`Retriever`, `reciprocal_rank_fusion`) |
| Reranking | `retrieval/rerank.py` (`LLMReranker`, `NoopReranker`) |

## Acceptance criteria
- [x] pgvector store passes the same behavioral tests as the in-memory store
- [x] Retriever supports `vector`, `keyword` and `hybrid` modes behind one setting
- [x] Fusion does not depend on raw score scales (RRF, k = 60)
- [x] Reranker failures never fail the request
- [x] Reranker token usage is reported for cost tracking

## Verification
17 new tests (38 total): RRF behavior, retriever modes and filters, reranker ordering, threshold, fallback on bad JSON and provider errors, out-of-range ids, cost tracking, and 3 integration tests against Postgres 16 with pgvector.

## Decisions and deviations from the plan
- **Keyword queries in Postgres OR the stemmed lexemes** of the question (`tsvector_to_array` → `tsquery`). `plainto_tsquery`/`websearch_to_tsquery` AND all terms, which returns nothing for most full-sentence questions.
- **Vectors are passed as pgvector literals** rather than through an extra adapter dependency.
- **Retries rely on the provider SDKs** (exponential backoff, configurable attempts and timeout) instead of a separate retry library, which keeps one retry policy per provider.
- **Anthropic calls omit `temperature`**: the current SDK no longer accepts it for this call.

## Known limitations
- The LLM reranker adds one model call per query (configurable off).
- No query rewriting for very short or ambiguous questions.

## Next steps
- M3: answer generation with validated citations and the HTTP API.
