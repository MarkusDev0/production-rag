# production-rag

[![CI](https://github.com/USERNAME/production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/production-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A retrieval-augmented generation (RAG) service built for production use: hybrid search, reranking, answers with validated citations, explicit abstention when the documents don't contain the answer, and an evaluation harness to measure all of it.

## Why this exists

Most RAG demos work on the first ten questions and fail with real users. The common failure modes are predictable:

- **Exact terms get lost.** Pure vector search misses product codes, policy names and numbers that keyword search finds instantly.
- **Citations can't be trusted.** Models cite sources that don't exist or don't support the claim.
- **The system guesses instead of saying "I don't know."** Missing context turns into confident hallucination.
- **Nobody can tell whether a change made things better.** Without an eval set, chunking and prompt changes are judged by feel.

This project addresses each of these directly and keeps every component behind a small interface, so stores, embedders and LLM providers can be swapped without touching retrieval or API code.

## Features

- **Ingestion** of PDF (with page numbers), HTML (boilerplate stripped), Markdown and plain text, from uploads or URLs
- **Structure-aware chunking** that splits on paragraphs, then lines, then sentences, with configurable overlap
- **Idempotent re-ingestion**: the same source replaces its previous version instead of duplicating it
- **Hybrid retrieval**: vector search + keyword search fused with Reciprocal Rank Fusion
- **LLM reranking** with graceful fallback to retrieval order if the call or parsing fails
- **Metadata filters** (e.g. `{"department": "finance"}`) applied inside the database query
- **Grounded answers**: numbered sources, `[n]` citations validated against what was actually retrieved, invented references stripped
- **Abstention**: an explicit `abstained: true` response instead of a guess, with no LLM call at all when nothing is retrieved
- **Cost and token tracking** per request, across reranking and generation
- **Observability**: structured JSON logs, request IDs, per-stage latency
- **Providers**: OpenAI or Anthropic for generation; OpenAI or an offline embedder for embeddings
- **Storage**: PostgreSQL + pgvector (HNSW, full-text GIN, JSONB filters) or in-memory

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[PDF / HTML / text / URL] --> B[Loaders]
        B --> C[Recursive chunker]
        C --> D[Embedder]
        D --> E[(Postgres + pgvector)]
    end

    subgraph Query
        Q[Question] --> V[Vector search]
        Q --> K[Keyword search]
        V --> F[Reciprocal Rank Fusion]
        K --> F
        F --> R[LLM reranker]
        R --> G[Answer generator]
        G --> X{Citations valid?<br/>Context sufficient?}
        X --> Y[Answer + citations + usage]
    end

    E -.-> V
    E -.-> K
```

More detail in [docs/architecture.md](docs/architecture.md).

## Quickstart

**With Docker** (Postgres + API):

```bash
cp .env.example .env          # add OPENAI_API_KEY (or ANTHROPIC_API_KEY + RAG_LLM_PROVIDER=anthropic)
docker compose up --build
```

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
make install
docker compose up -d db
make run
```

Interactive API docs: http://localhost:8000/docs

## Usage

Upload a document, with optional metadata for filtering:

```bash
curl -F "file=@employee-handbook.pdf" \
     -F 'metadata={"department": "hr"}' \
     http://localhost:8000/documents
```

Or ingest a web page:

```bash
curl -X POST http://localhost:8000/documents/url \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/pricing", "metadata": {"department": "sales"}}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How many days can I work remotely?", "filters": {"department": "hr"}}'
```

Example response:

```json
{
  "answer": "Office-based employees can work remotely up to 3 days per week after their 6-month probation [1].",
  "abstained": false,
  "citations": [
    {
      "ref": 1,
      "chunk_id": "4f1c2a9e0b7d3c11:7",
      "document_id": "4f1c2a9e0b7d3c11",
      "source": "employee-handbook.pdf",
      "title": "Employee Handbook",
      "page": 12,
      "snippet": "Office-based employees may work remotely up to 3 days per week..."
    }
  ],
  "usage": { "input_tokens": 1843, "output_tokens": 41, "cost_usd": 0.000318 },
  "latency_ms": 1412.6
}
```

| Endpoint | Description |
|---|---|
| `POST /documents` | Upload a file (`.pdf`, `.html`, `.md`, `.txt`) with optional JSON metadata |
| `POST /documents/url` | Fetch and ingest a URL |
| `GET /documents` | List ingested documents |
| `DELETE /documents/{id}` | Remove a document and all its chunks |
| `POST /query` | Ask a question, with optional `top_k` and `filters` |
| `GET /health` | Liveness check |

## Configuration

All settings come from environment variables (see [.env.example](.env.example)).

| Variable | Default | Notes |
|---|---|---|
| `RAG_LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `RAG_LLM_MODEL` | `gpt-4.1-mini` | Any chat model of the chosen provider |
| `RAG_EMBEDDING_PROVIDER` | `openai` | `hashing` runs offline (lexical only, for dev/tests) |
| `RAG_VECTOR_STORE` | `pgvector` | `memory` for quick experiments |
| `RAG_RETRIEVAL_MODE` | `hybrid` | `vector`, `keyword` or `hybrid` |
| `RAG_RERANKER` | `llm` | `none` to skip the extra LLM call |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` | `5` / `20` | Final results / candidates before reranking |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Characters |
| `RAG_PRICE_INPUT_PER_MTOK` / `RAG_PRICE_OUTPUT_PER_MTOK` | unset | Enables `cost_usd` in responses |

## Evaluation

The harness in [`evals/`](evals/) runs a labeled question set against a small document corpus. Each answerable question has an evidence span; unanswerable questions check that the system abstains.

```bash
make evals                                              # retrieval metrics, offline
python -m evals.run_evals --embedder openai             # with real embeddings
python -m evals.run_evals --generate --reranker llm     # full answer metrics (uses the LLM)
```

Offline baseline (hashing embedder, k = 3, 21 answerable + 3 unanswerable questions):

| mode | hit@3 | MRR@3 |
|---|---|---|
| vector | 0.905 | 0.810 |
| keyword | 0.952 | 0.881 |
| hybrid | 0.952 | 0.833 |

How to read this: the offline embedder is lexical, so vector and keyword search see the same signal and hybrid can't show its real benefit here. Hybrid search pays off with semantic embeddings, where vector search catches paraphrases and keyword search catches exact terms. The remaining miss ("How long do I have to submit an expense claim?") comes from "submit" not matching "submitted", which is a stemming gap tracked in the roadmap below.

CI runs the offline retrieval eval as a quality gate (`--min-hit-rate 0.85`), so a regression in chunking or retrieval fails the build.

## Design decisions

**Reciprocal Rank Fusion instead of weighted score blending.** Cosine similarity and BM25 / `ts_rank` scores live on unrelated scales. RRF uses ranks only, so it needs no per-corpus calibration.

**LLM reranker instead of a cross-encoder.** It keeps the deployment free of GPU and model-weight dependencies and works with whichever provider is configured. The cost is one extra call per query, so it can be turned off with `RAG_RERANKER=none`. A cross-encoder is a natural drop-in through the `Reranker` interface.

**Postgres + pgvector instead of a dedicated vector database.** One system holds vectors, full-text indexes, metadata and documents, with transactional upserts. For most business document collections (up to millions of chunks) this is simpler to operate, back up and secure.

**Abstention as a first-class result.** "I don't know" is returned with `abstained: true` so callers can route the question to a human, instead of parsing free text.

**Citations validated after generation.** The model is instructed to cite, but the code verifies every reference against the retrieved sources and removes any that don't exist.

**Fail soft where quality is optional, fail loud where correctness is required.** A failed reranker call falls back to retrieval order. A document with no extractable text is rejected with a clear error instead of being stored empty.

## Project structure

```
src/production_rag/
├── api/            FastAPI app, request/response schemas
├── ingestion/      loaders (PDF, HTML, text, URL) and chunking
├── retrieval/      hybrid retrieval (RRF) and rerankers
├── generation/     LLM clients and grounded answer generation
├── store/          storage interface, pgvector and in-memory backends
├── embeddings.py   embedding backends
├── pipeline.py     ingestion + query orchestration
├── factory.py      wiring from settings
└── config.py       typed settings
evals/              evaluation corpus, labeled questions and harness
tests/              unit, API and Postgres integration tests
docs/               architecture and milestone plans/reports
```

## Development

```bash
make lint          # ruff
make typecheck     # mypy --strict
make test          # unit + API tests
RAG_TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag make test-integration
```

Branching: `dev` for integration, `staging` for release candidates, `main` for releases. Milestone plans and delivery reports are in [docs/milestones](docs/milestones/).

## Limitations and roadmap

- **No OCR.** Scanned PDFs are rejected with a clear error; an OCR step (e.g. Tesseract or a vision model) is the next ingestion feature.
- **Keyword search in memory has no stemming.** Postgres full-text search stems; the in-memory BM25 doesn't yet.
- **Answer evaluation is keyword-based.** An LLM-as-judge faithfulness score would catch subtler errors.
- **No authentication or multi-tenancy.** Intended to run behind an API gateway; tenant isolation would use a `tenant_id` metadata filter enforced server-side.
- **Synchronous ingestion.** Large batches should move to a background queue.

## License

[MIT](LICENSE)
