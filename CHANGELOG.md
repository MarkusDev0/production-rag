# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.4.0] — Evaluation & hardening

### Added
- Evaluation harness with hit@k, MRR@k, answer accuracy, grounded citation rate and abstention metrics
- Evaluation corpus (6 policy documents) and 24 labeled questions, including 3 unanswerable ones
- CI workflow: lint, strict type checking, unit + Postgres integration tests on Python 3.11 and 3.12, retrieval quality gate, Docker build
- Architecture guide and full README

## [0.3.0] — Generation & API

### Added
- Grounded answer generation with numbered sources and citation validation
- Abstention when the sources don't contain the answer
- Token usage and cost tracking across reranking and generation
- RAG pipeline orchestrating ingestion and querying
- FastAPI service: upload, URL ingestion, list, delete, query, health
- Dockerfile and Docker Compose stack with pgvector

## [0.2.0] — Retrieval

### Added
- PostgreSQL + pgvector store with HNSW, full-text GIN and JSONB metadata indexes
- Provider-agnostic LLM client for OpenAI and Anthropic
- Hybrid retrieval with Reciprocal Rank Fusion
- LLM reranker with fallback to retrieval order

## [0.1.0] — Foundation & ingestion

### Added
- Typed settings, domain models and structured JSON logging
- PDF, HTML, text and URL loaders
- Recursive chunker with overlap and page tracking
- OpenAI and offline hashing embedders
- In-memory store with cosine similarity and BM25
