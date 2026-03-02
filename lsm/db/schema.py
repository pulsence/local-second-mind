"""Application schema ownership for the unified SQLite database.

This module is the single source of truth for all non-vector application
tables.  ``ensure_application_schema`` must be called once per database
lifetime (typically during provider initialization) before any
subsystem reads or writes.

Vector-specific DDL (``vec_chunks``, ``chunks_fts``, FTS triggers)
remains in ``lsm.vectordb.sqlite_vec``.
"""

from __future__ import annotations

import sqlite3

from .tables import DEFAULT_TABLE_NAMES, TableNames


def get_application_tables(prefix: str = "lsm_") -> tuple[str, ...]:
    """Return resolved application table names for the given prefix."""
    return TableNames(prefix=prefix).application_tables()


# Backward-compatible constant using default prefix.
APPLICATION_TABLES: tuple[str, ...] = DEFAULT_TABLE_NAMES.application_tables()


def ensure_application_schema(
    conn: sqlite3.Connection,
    table_names: TableNames | None = None,
) -> None:
    """Create all non-vector application tables and indexes.

    Safe to call multiple times (all statements use
    ``CREATE ... IF NOT EXISTS``).  Must be called **before** vector DDL
    so that FTS content-sync triggers can reference the chunks table.
    """
    tn = table_names or DEFAULT_TABLE_NAMES

    conn.executescript(
        f"""
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
    )
