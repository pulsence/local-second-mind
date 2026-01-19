"""
PostgreSQL + pgvector provider implementation.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Iterable, Tuple

from lsm.logging import get_logger
from lsm.config.models import VectorDBConfig
from .base import BaseVectorDBProvider, VectorDBQueryResult

logger = get_logger(__name__)

PSYCOPG2_AVAILABLE = False
PGVECTOR_AVAILABLE = False

try:
    import psycopg2  # noqa: F401
    from psycopg2 import pool  # noqa: F401
    from psycopg2 import sql  # noqa: F401
    from psycopg2.extras import execute_values  # noqa: F401
    PSYCOPG2_AVAILABLE = True
except Exception as e:
    logger.debug(f"psycopg2 not available: {e}")

try:
    from pgvector.psycopg2 import register_vector  # noqa: F401
    PGVECTOR_AVAILABLE = True
except Exception as e:
    logger.debug(f"pgvector not available: {e}")


class PostgreSQLProvider(BaseVectorDBProvider):
    """PostgreSQL + pgvector provider."""

    def __init__(self, config: VectorDBConfig) -> None:
        super().__init__(config)
        self._pool = None
        self._table_name = self._sanitize_table_name(config.collection or "local_kb")
        self._embedding_dim: Optional[int] = None

    @property
    def name(self) -> str:
        return "postgresql"

    @staticmethod
    def _sanitize_table_name(value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
        if not safe:
            safe = "local_kb"
        return f"chunks_{safe}"

    def _ensure_pool(self) -> None:
        if self._pool is not None:
            return

        if self.config.connection_string:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=max(1, int(self.config.pool_size)),
                dsn=self.config.connection_string,
            )
        else:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=max(1, int(self.config.pool_size)),
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
            )

    @contextmanager
    def _get_conn(self):
        self._ensure_pool()
        if not self._pool:
            raise RuntimeError("PostgreSQL connection pool not initialized")
        conn = self._pool.getconn()
        try:
            if PGVECTOR_AVAILABLE:
                register_vector(conn)
            yield conn
        finally:
            self._pool.putconn(conn)

    def _ensure_extension(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()

    def _ensure_schema(self, conn, embedding_dim: int) -> None:
        if self._embedding_dim is None:
            self._embedding_dim = embedding_dim

        table_ident = sql.Identifier(self._table_name)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        doc_id TEXT,
                        chunk_index INTEGER,
                        text TEXT NOT NULL,
                        embedding vector({dim}),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                    """
                ).format(table=table_ident, dim=sql.Literal(self._embedding_dim))
            )
        conn.commit()

    def _ensure_indexes(self, conn) -> None:
        table_ident = sql.Identifier(self._table_name)
        index_type = (self.config.index_type or "hnsw").lower()

        with conn.cursor() as cur:
            if index_type == "hnsw":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw (embedding vector_cosine_ops)"
                    ).format(
                        idx=sql.Identifier(f"{self._table_name}_emb_hnsw_idx"),
                        table=table_ident,
                    )
                )
            elif index_type == "ivfflat":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING ivfflat (embedding vector_cosine_ops)"
                    ).format(
                        idx=sql.Identifier(f"{self._table_name}_emb_ivfflat_idx"),
                        table=table_ident,
                    )
                )

            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} (doc_id)"
                ).format(
                    idx=sql.Identifier(f"{self._table_name}_doc_id_idx"),
                    table=table_ident,
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING gin (metadata)"
                ).format(
                    idx=sql.Identifier(f"{self._table_name}_metadata_gin_idx"),
                    table=table_ident,
                )
            )
        conn.commit()

    def _table_exists(self, conn) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", [self._table_name])
            return cur.fetchone()[0] is not None

    def is_available(self) -> bool:
        if not (PSYCOPG2_AVAILABLE and PGVECTOR_AVAILABLE):
            return False
        try:
            self._ensure_pool()
            return True
        except Exception as e:
            logger.debug(f"PostgreSQL provider unavailable: {e}")
            return False

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if not ids:
            return

        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas, and embeddings must have the same length")

        embedding_dim = len(embeddings[0]) if embeddings else 0
        if embedding_dim <= 0:
            raise ValueError("Embeddings must be non-empty to infer dimension")

        with self._get_conn() as conn:
            self._ensure_extension(conn)
            self._ensure_schema(conn, embedding_dim)
            self._ensure_indexes(conn)

            rows: List[Tuple[str, Optional[str], Optional[int], str, List[float], str]] = []
            for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                doc_id = meta.get("doc_id") if isinstance(meta, dict) else None
                chunk_index = meta.get("chunk_index") if isinstance(meta, dict) else None
                rows.append(
                    (
                        str(cid),
                        str(doc_id) if doc_id is not None else None,
                        int(chunk_index) if chunk_index is not None else None,
                        doc or "",
                        emb,
                        json.dumps(meta or {}),
                    )
                )

            table_ident = sql.Identifier(self._table_name)
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    sql.SQL(
                        """
                        INSERT INTO {table} (id, doc_id, chunk_index, text, embedding, metadata)
                        VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            chunk_index = EXCLUDED.chunk_index,
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata
                        """
                    ).format(table=table_ident),
                    rows,
                )
            conn.commit()

    def query(
        self,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> VectorDBQueryResult:
        if not embedding:
            return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

        with self._get_conn() as conn:
            if not self._table_exists(conn):
                return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])
            self._ensure_extension(conn)
            self._ensure_schema(conn, len(embedding))
            self._ensure_indexes(conn)

            table_ident = sql.Identifier(self._table_name)
            where_sql = sql.SQL("")
            params: List[Any] = [embedding, embedding, top_k]

            if filters:
                where_sql = sql.SQL("WHERE metadata @> %s::jsonb")
                params = [embedding, json.dumps(filters), embedding, top_k]

            query_sql = sql.SQL(
                """
                SELECT id, text, metadata, (embedding <=> %s) AS distance
                FROM {table}
                {where_clause}
                ORDER BY embedding <=> %s
                LIMIT %s
                """
            ).format(table=table_ident, where_clause=where_sql)

            with conn.cursor() as cur:
                cur.execute(query_sql, params)
                rows = cur.fetchall()

        ids = [row[0] for row in rows]
        documents = [row[1] or "" for row in rows]
        metadatas = [row[2] or {} for row in rows]
        distances = [float(row[3]) if row[3] is not None else None for row in rows]

        return VectorDBQueryResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
        )

    def delete_by_id(self, ids: List[str]) -> None:
        if not ids:
            return
        with self._get_conn() as conn:
            if not self._table_exists(conn):
                return
            table_ident = sql.Identifier(self._table_name)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(table=table_ident),
                    [ids],
                )
            conn.commit()

    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        if not filters:
            raise ValueError("filters must be a non-empty dict")
        with self._get_conn() as conn:
            if not self._table_exists(conn):
                return
            table_ident = sql.Identifier(self._table_name)
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DELETE FROM {table} WHERE metadata @> %s::jsonb").format(
                        table=table_ident
                    ),
                    [json.dumps(filters)],
                )
            conn.commit()

    def count(self) -> int:
        with self._get_conn() as conn:
            if not self._table_exists(conn):
                return 0
            table_ident = sql.Identifier(self._table_name)
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {table}").format(table=table_ident))
                value = cur.fetchone()[0]
        return int(value)

    def get_stats(self) -> Dict[str, Any]:
        stats = {"provider": self.name, "table": self._table_name}
        try:
            stats["count"] = self.count()
        except Exception as e:
            stats["error"] = str(e)
        return stats

    def optimize(self) -> Dict[str, Any]:
        with self._get_conn() as conn:
            if not self._table_exists(conn):
                return {"provider": self.name, "status": "empty"}
            table_ident = sql.Identifier(self._table_name)
            with conn.cursor() as cur:
                cur.execute(sql.SQL("VACUUM ANALYZE {table}").format(table=table_ident))
            conn.commit()
        return {"provider": self.name, "status": "ok"}

    def health_check(self) -> Dict[str, Any]:
        if not self.is_available():
            return {
                "provider": self.name,
                "status": "missing_dependencies",
                "psycopg2": PSYCOPG2_AVAILABLE,
                "pgvector": PGVECTOR_AVAILABLE,
            }
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return {"provider": self.name, "status": "ok"}
        except Exception as e:
            return {"provider": self.name, "status": "error", "error": str(e)}
