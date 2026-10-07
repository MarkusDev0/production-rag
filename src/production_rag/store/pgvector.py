"""PostgreSQL + pgvector store.

- Vector search: cosine distance over an HNSW index.
- Keyword search: Postgres full-text search (OR-ed stemmed terms ranked with `ts_rank_cd`)
  over a generated `tsvector` column with a GIN index.
- Metadata filters: JSONB containment (`@>`) with a GIN index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from production_rag.models import Chunk, Document, ScoredChunk
from production_rag.store.base import Filters


def _to_vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def _row_to_scored(row: Any) -> ScoredChunk:  # dict row from psycopg's dict_row factory
    chunk = Chunk(
        id=row["id"],
        document_id=row["document_id"],
        index=row["chunk_index"],
        text=row["text"],
        metadata=row["metadata"] or {},
    )
    return ScoredChunk(chunk=chunk, score=float(row["score"]))


class PgVectorStore:
    def __init__(
        self, database_url: str, embedding_dim: int, *, min_size: int = 1, max_size: int = 10
    ):
        self.embedding_dim = int(embedding_dim)
        self._pool = ConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        self.migrate()

    def close(self) -> None:
        self._pool.close()

    def migrate(self) -> None:
        dim = self.embedding_dim
        statements = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                source      TEXT NOT NULL,
                title       TEXT NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{}',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                text        TEXT NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{{}}',
                embedding   vector({dim}) NOT NULL,
                tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            )
            """,
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)",
            "CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv)",
            "CREATE INDEX IF NOT EXISTS chunks_metadata_gin ON chunks USING gin (metadata)",
            "CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id)",
        ]
        with self._pool.connection() as conn:
            for statement in statements:
                conn.execute(statement)

    def upsert(
        self, document: Document, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        with self._pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM documents WHERE id = %s", (document.id,))
            conn.execute(
                "INSERT INTO documents (id, source, title, metadata) VALUES (%s, %s, %s, %s)",
                (document.id, document.source, document.title, Jsonb(document.metadata)),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (id, document_id, chunk_index, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    [
                        (
                            c.id,
                            c.document_id,
                            c.index,
                            c.text,
                            Jsonb(c.metadata),
                            _to_vector_literal(v),
                        )
                        for c, v in zip(chunks, embeddings, strict=True)
                    ],
                )

    def delete_document(self, document_id: str) -> bool:
        with self._pool.connection() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            return cur.rowcount > 0

    def list_documents(self) -> list[Document]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, source, title, metadata FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [Document.model_validate(row) for row in rows]

    def vector_search(
        self, embedding: Sequence[float], k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        literal = _to_vector_literal(embedding)
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, text, metadata,
                       1 - (embedding <=> %(q)s::vector) AS score
                FROM chunks
                WHERE metadata @> %(filters)s
                ORDER BY embedding <=> %(q)s::vector
                LIMIT %(k)s
                """,
                {"q": literal, "filters": Jsonb(filters or {}), "k": k},
            ).fetchall()
        return [_row_to_scored(r) for r in rows]

    def keyword_search(
        self, query: str, k: int, filters: Filters | None = None
    ) -> list[ScoredChunk]:
        # OR the terms together: a plain AND query returns nothing for most natural questions.
        terms = [t for t in query.replace('"', " ").split() if t.strip()]
        if not terms:
            return []
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                WITH q AS (
                    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
                    FROM unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS lexeme
                )
                SELECT c.id, c.document_id, c.chunk_index, c.text, c.metadata,
                       ts_rank_cd(c.tsv, q.query) AS score
                FROM chunks c, q
                WHERE q.query IS NOT NULL AND c.tsv @@ q.query AND c.metadata @> %(filters)s
                ORDER BY score DESC
                LIMIT %(k)s
                """,
                {"text": " ".join(terms), "filters": Jsonb(filters or {}), "k": k},
            ).fetchall()
        return [_row_to_scored(r) for r in rows]
