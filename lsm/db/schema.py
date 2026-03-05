"""Application schema ownership for the unified database.

This module is the single source of truth for all non-vector application
tables.  ``ensure_application_schema`` must be called once per database
lifetime (typically during provider initialization) before any
subsystem reads or writes.

Supports both SQLite and PostgreSQL backends.  Dialect-specific type
differences (INTEGER vs BIGINT, REAL vs DOUBLE PRECISION, BLOB vs BYTEA,
AUTOINCREMENT vs SERIAL) are normalised per connection.

**Chunks table architecture decision**: The application-layer ``chunks``
table uses normalised columns on *both* backends for consistent access.
The PostgreSQL vector provider maintains a separate ``chunks_{collection}``
table with JSONB metadata for vector operations.  This divergence is
intentional — the application ``chunks`` table provides fast column-level
queries while the vector table supports flexible metadata filtering.

Vector-specific DDL (``vec_chunks``, ``chunks_fts``, FTS triggers)
remains in ``lsm.vectordb.sqlite_vec``.
"""

from __future__ import annotations

from typing import Any

from .compat import execute_ddl_script, is_sqlite
from .tables import DEFAULT_TABLE_NAMES, TableNames


def get_application_tables(prefix: str = "lsm_") -> tuple[str, ...]:
    """Return resolved application table names for the given prefix."""
    return TableNames(prefix=prefix).application_tables()


# Backward-compatible constant using default prefix.
APPLICATION_TABLES: tuple[str, ...] = DEFAULT_TABLE_NAMES.application_tables()


def _sqlite_ddl(tn: TableNames) -> str:
    """Return the full DDL script for SQLite."""
    return f"""
        CREATE TABLE IF NOT EXISTS {tn.chunks} (
            chunk_id         TEXT PRIMARY KEY,
            source_path      TEXT NOT NULL,
            source_name      TEXT,
            chunk_text       TEXT NOT NULL,
            heading          TEXT,
            heading_path     TEXT,
            page_number      TEXT,
            paragraph_index  INTEGER,
            mtime_ns         INTEGER,
            file_hash        TEXT,
            version          INTEGER,
            is_current       INTEGER DEFAULT 1,
            node_type        TEXT DEFAULT 'chunk',
            root_tags        TEXT,
            folder_tags      TEXT,
            content_type     TEXT,
            cluster_id       INTEGER,
            cluster_size     INTEGER,
            simhash          INTEGER,
            ext              TEXT,
            chunk_index      INTEGER,
            ingested_at      TEXT,
            start_char       INTEGER,
            end_char         INTEGER,
            chunk_length     INTEGER,
            metadata_json    TEXT NOT NULL DEFAULT '{{}}'
        );

        CREATE TABLE IF NOT EXISTS {tn.schema_versions} (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            manifest_version INTEGER,
            lsm_version      TEXT,
            embedding_model  TEXT,
            embedding_dim    INTEGER,
            chunking_strategy TEXT,
            chunk_size       INTEGER,
            chunk_overlap    INTEGER,
            created_at       TEXT,
            last_ingest_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.manifest} (
            source_path      TEXT PRIMARY KEY,
            mtime_ns         INTEGER,
            file_size        INTEGER,
            file_hash        TEXT,
            version          INTEGER,
            embedding_model  TEXT,
            schema_version_id INTEGER REFERENCES {tn.schema_versions}(id),
            updated_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.reranker_cache} (
            cache_key        TEXT PRIMARY KEY,
            score            REAL,
            created_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_memories} (
            id               TEXT PRIMARY KEY,
            memory_type      TEXT NOT NULL,
            memory_key       TEXT NOT NULL,
            value_json       TEXT NOT NULL,
            scope            TEXT NOT NULL,
            tags_json        TEXT NOT NULL,
            confidence       REAL NOT NULL,
            created_at       TEXT NOT NULL,
            last_used_at     TEXT NOT NULL,
            expires_at       TEXT NULL,
            source_run_id    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_memory_candidates} (
            id               TEXT PRIMARY KEY,
            memory_id        TEXT NOT NULL UNIQUE,
            provenance       TEXT NOT NULL,
            rationale        TEXT NOT NULL,
            status           TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES {tn.agent_memories}(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_schedules} (
            schedule_id      TEXT PRIMARY KEY,
            agent_name       TEXT NOT NULL,
            last_run_at      TEXT,
            next_run_at      TEXT NOT NULL,
            last_status      TEXT DEFAULT 'idle',
            last_error       TEXT,
            queued_runs      INTEGER DEFAULT 0,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.cluster_centroids} (
            cluster_id       INTEGER PRIMARY KEY,
            centroid         BLOB,
            size             INTEGER
        );

        CREATE TABLE IF NOT EXISTS {tn.graph_nodes} (
            node_id          TEXT PRIMARY KEY,
            node_type        TEXT,
            label            TEXT,
            source_path      TEXT,
            heading_path     TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.graph_edges} (
            edge_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id           TEXT REFERENCES {tn.graph_nodes}(node_id),
            dst_id           TEXT REFERENCES {tn.graph_nodes}(node_id),
            edge_type        TEXT,
            weight           REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS {tn.embedding_models} (
            model_id         TEXT PRIMARY KEY,
            base_model       TEXT,
            path             TEXT,
            dimension        INTEGER,
            created_at       TEXT,
            is_active        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS {tn.job_status} (
            job_name         TEXT PRIMARY KEY,
            status           TEXT NOT NULL,
            started_at       TEXT,
            completed_at     TEXT,
            corpus_size      INTEGER,
            metadata         TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.stats_cache} (
            cache_key        TEXT PRIMARY KEY,
            cached_at        REAL NOT NULL,
            chunk_count      INTEGER NOT NULL,
            stats_json       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.remote_cache} (
            cache_key        TEXT PRIMARY KEY,
            provider         TEXT NOT NULL,
            response_json    TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            expires_at       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_source_path ON {tn.chunks}(source_path);
        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_is_current ON {tn.chunks}(is_current);
        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_ext ON {tn.chunks}(ext);
        CREATE INDEX IF NOT EXISTS idx_{tn.manifest}_updated_at ON {tn.manifest}(updated_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memory_candidates}_status ON {tn.agent_memory_candidates}(status);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memories}_scope_type ON {tn.agent_memories}(scope, memory_type);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memories}_expires_at ON {tn.agent_memories}(expires_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_schedules}_next_run ON {tn.agent_schedules}(next_run_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_nodes}_source_path ON {tn.graph_nodes}(source_path);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_edges}_src ON {tn.graph_edges}(src_id);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_edges}_dst ON {tn.graph_edges}(dst_id);
        CREATE INDEX IF NOT EXISTS idx_{tn.stats_cache}_cached_at ON {tn.stats_cache}(cached_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.remote_cache}_provider ON {tn.remote_cache}(provider);
        CREATE INDEX IF NOT EXISTS idx_{tn.remote_cache}_expires_at ON {tn.remote_cache}(expires_at);
    """


def _postgres_ddl(tn: TableNames) -> str:
    """Return the full DDL script for PostgreSQL.

    Type differences from SQLite:
    - INTEGER -> BIGINT for large numeric columns (mtime_ns, file_size)
    - REAL -> DOUBLE PRECISION for confidence/weight/score
    - BLOB -> BYTEA for binary data (centroids)
    - INTEGER PRIMARY KEY AUTOINCREMENT -> SERIAL PRIMARY KEY
    - Foreign keys use same syntax (PostgreSQL supports it natively)
    """
    return f"""
        CREATE TABLE IF NOT EXISTS {tn.chunks} (
            chunk_id         TEXT PRIMARY KEY,
            source_path      TEXT NOT NULL,
            source_name      TEXT,
            chunk_text       TEXT NOT NULL,
            heading          TEXT,
            heading_path     TEXT,
            page_number      TEXT,
            paragraph_index  INTEGER,
            mtime_ns         BIGINT,
            file_hash        TEXT,
            version          INTEGER,
            is_current       INTEGER DEFAULT 1,
            node_type        TEXT DEFAULT 'chunk',
            root_tags        TEXT,
            folder_tags      TEXT,
            content_type     TEXT,
            cluster_id       INTEGER,
            cluster_size     INTEGER,
            simhash          BIGINT,
            ext              TEXT,
            chunk_index      INTEGER,
            ingested_at      TEXT,
            start_char       INTEGER,
            end_char         INTEGER,
            chunk_length     INTEGER,
            metadata_json    TEXT NOT NULL DEFAULT '{{}}'
        );

        CREATE TABLE IF NOT EXISTS {tn.schema_versions} (
            id               SERIAL PRIMARY KEY,
            manifest_version BIGINT,
            lsm_version      TEXT,
            embedding_model  TEXT,
            embedding_dim    BIGINT,
            chunking_strategy TEXT,
            chunk_size       BIGINT,
            chunk_overlap    BIGINT,
            created_at       TEXT,
            last_ingest_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.manifest} (
            source_path      TEXT PRIMARY KEY,
            mtime_ns         BIGINT,
            file_size         BIGINT,
            file_hash        TEXT,
            version          BIGINT,
            embedding_model  TEXT,
            schema_version_id BIGINT REFERENCES {tn.schema_versions}(id),
            updated_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.reranker_cache} (
            cache_key        TEXT PRIMARY KEY,
            score            DOUBLE PRECISION,
            created_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_memories} (
            id               TEXT PRIMARY KEY,
            memory_type      TEXT NOT NULL,
            memory_key       TEXT NOT NULL,
            value_json       TEXT NOT NULL,
            scope            TEXT NOT NULL,
            tags_json        TEXT NOT NULL,
            confidence       DOUBLE PRECISION NOT NULL,
            created_at       TEXT NOT NULL,
            last_used_at     TEXT NOT NULL,
            expires_at       TEXT NULL,
            source_run_id    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_memory_candidates} (
            id               TEXT PRIMARY KEY,
            memory_id        TEXT NOT NULL UNIQUE,
            provenance       TEXT NOT NULL,
            rationale        TEXT NOT NULL,
            status           TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES {tn.agent_memories}(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS {tn.agent_schedules} (
            schedule_id      TEXT PRIMARY KEY,
            agent_name       TEXT NOT NULL,
            last_run_at      TEXT,
            next_run_at      TEXT NOT NULL,
            last_status      TEXT DEFAULT 'idle',
            last_error       TEXT,
            queued_runs      BIGINT DEFAULT 0,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.cluster_centroids} (
            cluster_id       INTEGER PRIMARY KEY,
            centroid         BYTEA,
            size             INTEGER
        );

        CREATE TABLE IF NOT EXISTS {tn.graph_nodes} (
            node_id          TEXT PRIMARY KEY,
            node_type        TEXT,
            label            TEXT,
            source_path      TEXT,
            heading_path     TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.graph_edges} (
            edge_id          SERIAL PRIMARY KEY,
            src_id           TEXT REFERENCES {tn.graph_nodes}(node_id),
            dst_id           TEXT REFERENCES {tn.graph_nodes}(node_id),
            edge_type        TEXT,
            weight           DOUBLE PRECISION DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS {tn.embedding_models} (
            model_id         TEXT PRIMARY KEY,
            base_model       TEXT,
            path             TEXT,
            dimension        INTEGER,
            created_at       TEXT,
            is_active        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS {tn.job_status} (
            job_name         TEXT PRIMARY KEY,
            status           TEXT NOT NULL,
            started_at       TEXT,
            completed_at     TEXT,
            corpus_size      BIGINT,
            metadata         TEXT
        );

        CREATE TABLE IF NOT EXISTS {tn.stats_cache} (
            cache_key        TEXT PRIMARY KEY,
            cached_at        DOUBLE PRECISION NOT NULL,
            chunk_count      BIGINT NOT NULL,
            stats_json       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS {tn.remote_cache} (
            cache_key        TEXT PRIMARY KEY,
            provider         TEXT NOT NULL,
            response_json    TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            expires_at       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_source_path ON {tn.chunks}(source_path);
        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_is_current ON {tn.chunks}(is_current);
        CREATE INDEX IF NOT EXISTS idx_{tn.chunks}_ext ON {tn.chunks}(ext);
        CREATE INDEX IF NOT EXISTS idx_{tn.manifest}_updated_at ON {tn.manifest}(updated_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memory_candidates}_status ON {tn.agent_memory_candidates}(status);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memories}_scope_type ON {tn.agent_memories}(scope, memory_type);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_memories}_expires_at ON {tn.agent_memories}(expires_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.agent_schedules}_next_run ON {tn.agent_schedules}(next_run_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_nodes}_source_path ON {tn.graph_nodes}(source_path);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_edges}_src ON {tn.graph_edges}(src_id);
        CREATE INDEX IF NOT EXISTS idx_{tn.graph_edges}_dst ON {tn.graph_edges}(dst_id);
        CREATE INDEX IF NOT EXISTS idx_{tn.stats_cache}_cached_at ON {tn.stats_cache}(cached_at);
        CREATE INDEX IF NOT EXISTS idx_{tn.remote_cache}_provider ON {tn.remote_cache}(provider);
        CREATE INDEX IF NOT EXISTS idx_{tn.remote_cache}_expires_at ON {tn.remote_cache}(expires_at);
    """


def ensure_application_schema(
    conn: Any,
    table_names: TableNames | None = None,
) -> None:
    """Create all non-vector application tables and indexes.

    Supports both SQLite and PostgreSQL connections.  Safe to call
    multiple times (all statements use ``CREATE ... IF NOT EXISTS``).
    Must be called **before** vector DDL so that FTS content-sync
    triggers can reference the chunks table.

    Args:
        conn: A ``sqlite3.Connection`` or ``psycopg2`` connection.
        table_names: Optional ``TableNames`` override.
    """
    tn = table_names or DEFAULT_TABLE_NAMES

    if is_sqlite(conn):
        ddl = _sqlite_ddl(tn)
    else:
        ddl = _postgres_ddl(tn)

    execute_ddl_script(conn, ddl)
