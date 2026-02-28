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

# Canonical list of application tables created by ensure_application_schema.
# Used by health-check routines and migration tooling.
APPLICATION_TABLES: tuple[str, ...] = (
    "lsm_chunks",
    "lsm_schema_versions",
    "lsm_manifest",
    "lsm_reranker_cache",
    "lsm_agent_memories",
    "lsm_agent_memory_candidates",
    "lsm_agent_schedules",
    "lsm_cluster_centroids",
    "lsm_graph_nodes",
    "lsm_graph_edges",
    "lsm_embedding_models",
    "lsm_job_status",
    "lsm_stats_cache",
    "lsm_remote_cache",
)


def ensure_application_schema(conn: sqlite3.Connection) -> None:
    """Create all non-vector application tables and indexes.

    Safe to call multiple times (all statements use
    ``CREATE … IF NOT EXISTS``).  Must be called **before** vector DDL
    so that FTS content-sync triggers can reference ``lsm_chunks``.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS lsm_chunks (
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
            metadata_json    TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS lsm_schema_versions (
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

        CREATE TABLE IF NOT EXISTS lsm_manifest (
            source_path      TEXT PRIMARY KEY,
            mtime_ns         INTEGER,
            file_size        INTEGER,
            file_hash        TEXT,
            version          INTEGER,
            embedding_model  TEXT,
            schema_version_id INTEGER REFERENCES lsm_schema_versions(id),
            updated_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS lsm_reranker_cache (
            cache_key        TEXT PRIMARY KEY,
            score            REAL,
            created_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS lsm_agent_memories (
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

        CREATE TABLE IF NOT EXISTS lsm_agent_memory_candidates (
            id               TEXT PRIMARY KEY,
            memory_id        TEXT NOT NULL UNIQUE,
            provenance       TEXT NOT NULL,
            rationale        TEXT NOT NULL,
            status           TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            FOREIGN KEY(memory_id) REFERENCES lsm_agent_memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS lsm_agent_schedules (
            schedule_id      TEXT PRIMARY KEY,
            agent_name       TEXT NOT NULL,
            last_run_at      TEXT,
            next_run_at      TEXT NOT NULL,
            last_status      TEXT DEFAULT 'idle',
            last_error       TEXT,
            queued_runs      INTEGER DEFAULT 0,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lsm_cluster_centroids (
            cluster_id       INTEGER PRIMARY KEY,
            centroid         BLOB,
            size             INTEGER
        );

        CREATE TABLE IF NOT EXISTS lsm_graph_nodes (
            node_id          TEXT PRIMARY KEY,
            node_type        TEXT,
            label            TEXT,
            source_path      TEXT,
            heading_path     TEXT
        );

        CREATE TABLE IF NOT EXISTS lsm_graph_edges (
            edge_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id           TEXT REFERENCES lsm_graph_nodes(node_id),
            dst_id           TEXT REFERENCES lsm_graph_nodes(node_id),
            edge_type        TEXT,
            weight           REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS lsm_embedding_models (
            model_id         TEXT PRIMARY KEY,
            base_model       TEXT,
            path             TEXT,
            dimension        INTEGER,
            created_at       TEXT,
            is_active        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS lsm_job_status (
            job_name         TEXT PRIMARY KEY,
            status           TEXT NOT NULL,
            started_at       TEXT,
            completed_at     TEXT,
            corpus_size      INTEGER,
            metadata         TEXT
        );

        CREATE TABLE IF NOT EXISTS lsm_stats_cache (
            cache_key        TEXT PRIMARY KEY,
            cached_at        REAL NOT NULL,
            chunk_count      INTEGER NOT NULL,
            stats_json       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS lsm_remote_cache (
            cache_key        TEXT PRIMARY KEY,
            provider         TEXT NOT NULL,
            response_json    TEXT NOT NULL,
            created_at       TEXT NOT NULL,
            expires_at       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_lsm_chunks_source_path ON lsm_chunks(source_path);
        CREATE INDEX IF NOT EXISTS idx_lsm_chunks_is_current ON lsm_chunks(is_current);
        CREATE INDEX IF NOT EXISTS idx_lsm_chunks_ext ON lsm_chunks(ext);
        CREATE INDEX IF NOT EXISTS idx_lsm_manifest_updated_at ON lsm_manifest(updated_at);
        CREATE INDEX IF NOT EXISTS idx_lsm_agent_memory_candidates_status ON lsm_agent_memory_candidates(status);
        CREATE INDEX IF NOT EXISTS idx_lsm_agent_memories_scope_type ON lsm_agent_memories(scope, memory_type);
        CREATE INDEX IF NOT EXISTS idx_lsm_agent_memories_expires_at ON lsm_agent_memories(expires_at);
        CREATE INDEX IF NOT EXISTS idx_lsm_agent_schedules_next_run ON lsm_agent_schedules(next_run_at);
        CREATE INDEX IF NOT EXISTS idx_lsm_graph_edges_src ON lsm_graph_edges(src_id);
        CREATE INDEX IF NOT EXISTS idx_lsm_graph_edges_dst ON lsm_graph_edges(dst_id);
        CREATE INDEX IF NOT EXISTS idx_lsm_stats_cache_cached_at ON lsm_stats_cache(cached_at);
        CREATE INDEX IF NOT EXISTS idx_lsm_remote_cache_provider ON lsm_remote_cache(provider);
        CREATE INDEX IF NOT EXISTS idx_lsm_remote_cache_expires_at ON lsm_remote_cache(expires_at);
        """
    )
