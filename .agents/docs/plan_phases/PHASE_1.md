# Phase 1: SQLite-vec Provider and Unified Schema

**Status**: Pending

Replaces ChromaDB with sqlite-vec as the default vector store. Creates the unified
`lsm.db` database file containing all LSM state tables. Updates the `VectorDBConfig`
model and factory. This phase establishes the foundation that every subsequent phase
depends on.

Reference: [RESEARCH_PLAN.md §3.1, §3.2, §3.8](../RESEARCH_PLAN.md#3-part-a-unified-database-architecture)

---

## 1.1: VectorDBConfig Breaking Changes

**Description**: Simplify the `VectorDBConfig` dataclass for v0.8.0. Remove ChromaDB-specific
fields, change the default provider from `"chromadb"` to `"sqlite"`, and rename `persist_dir`
to `path`.

**Tasks**:
- Update `VectorDBConfig` in `lsm/config/models/vectordb.py`:
  - Change `provider` default from `"chromadb"` to `"sqlite"`
  - Rename `persist_dir` to `path` (directory where `lsm.db` lives)
  - Remove `chroma_hnsw_space` field
  - Remove `chroma_flush_interval` field (also remove from `IngestConfig` if present)
  - Remove `enable_versioning` from `IngestConfig` (versioning is always on — §3.5)
  - Remove `ingest.manifest` config field (manifest is DB-only)
  - Retain `collection` field (table name prefix, optional)
- Update `example_config.json` to reflect the new `vectordb` section format
- Update config loader (`lsm/config/loader.py`) to reject old field names with a clear
  error message directing users to update their config
- Update all test fixtures (`conftest.py`, inline config dicts) that reference old fields
- Run: `pytest tests/test_config/ -v`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/vectordb.py` — field changes
- `lsm/config/models/ingest.py` — remove `enable_versioning`, `manifest`, `chroma_flush_interval`
- `lsm/config/loader.py` — rejection of legacy fields
- `example_config.json` — updated vectordb section
- `tests/test_config/` — fixture updates

**Success criteria**: `VectorDBConfig` loads with new field names. Old field names produce
a clear error. `enable_versioning` and `manifest` are gone from `IngestConfig`. All config
tests pass.

---

## 1.2: SQLiteVecProvider — Core Implementation

**Description**: Create the new `SQLiteVecProvider` class implementing `BaseVectorDBProvider`.
This is the primary vector store for v0.8.0. Uses sqlite-vec for vector search and FTS5 for
full-text search, both in a single `lsm.db` file.

**Tasks**:
- Add `sqlite-vec` to project dependencies in `pyproject.toml`
- Create `lsm/vectordb/sqlite_vec.py` with class `SQLiteVecProvider(BaseVectorDBProvider)`:
  - `__init__(config: VectorDBConfig)`: Open/create `lsm.db` at `config.path`,
    load sqlite-vec extension, set `journal_mode=WAL`, `busy_timeout=5000`
  - `_ensure_schema()`: Create all tables from §3.2 if they don't exist:
    - `lsm_chunks` (core metadata table)
    - `vec_chunks` (vec0 virtual table for vector search)
    - `chunks_fts` (FTS5 virtual table for full-text search)
    - `lsm_manifest` (file-level ingest tracking)
    - `lsm_schema_versions` (schema version tracking)
    - `lsm_reranker_cache` (cross-encoder cache)
    - `lsm_agent_memories` (agent memory — replaces memories.db)
    - `lsm_agent_memory_candidates` (memory candidates)
    - `lsm_agent_schedules` (schedule state — replaces schedules.json)
    - `lsm_cluster_centroids` (Phase 2 — created but empty)
    - `lsm_graph_nodes`, `lsm_graph_edges` (Phase 3 — created but empty)
    - `lsm_embedding_models` (Phase 3 — created but empty)
    - `lsm_job_status` (offline job tracking)
    - All indexes from §3.2
  - Implement `BaseVectorDBProvider` interface methods:
    - `add_chunks(ids, embeddings, metadatas, texts)` — INSERT into `lsm_chunks` +
      `vec_chunks` + `chunks_fts` in a single transaction
    - `get(ids)` — SELECT from `lsm_chunks`
    - `query(embedding, top_k, filters)` — KNN via `vec_chunks` with metadata pre-filters
    - `delete_by_id(ids)` — DELETE from all three tables
    - `delete_by_filter(filters)` — DELETE by metadata filter
    - `delete_all()` — TRUNCATE all chunk tables
    - `count()` — `SELECT COUNT(*) FROM lsm_chunks WHERE is_current = 1`
    - `get_stats()` — aggregate statistics
    - `optimize()` — `VACUUM` + `ANALYZE`
    - `health_check()` — verify sqlite-vec extension loaded, tables exist
    - `update_metadatas(ids, metadatas)` — UPDATE `lsm_chunks`
  - Expose `connection` property for direct SQL access by other subsystems
    (agent memory, scheduler, manifest)
  - Use parameterized queries for all SQL (no f-strings with user data)
- Write comprehensive tests

- Commit and push changes for this sub-phase.
**Files**:
- `pyproject.toml` — add `sqlite-vec` dependency
- `lsm/vectordb/sqlite_vec.py` — new provider implementation
- `tests/test_vectordb/test_sqlite_vec.py` — unit + integration tests:
  - Test: schema creation on fresh database
  - Test: add_chunks inserts into all three tables atomically
  - Test: query returns correct KNN results (distance ordering)
  - Test: metadata filters work (path_contains, ext_allow, ext_deny, is_current)
  - Test: delete_by_id removes from all tables
  - Test: FTS5 content-sync table stays consistent with lsm_chunks
  - Test: WAL mode is set
  - Test: concurrent reads don't block
  - Test: health_check passes with valid DB, fails with missing extension

**Success criteria**: `SQLiteVecProvider` passes all `BaseVectorDBProvider` interface tests.
A single `lsm.db` file contains vectors, metadata, and FTS5 index. KNN query returns
correct results with cosine distance.

---

## 1.3: VectorDB Factory and ChromaDB Removal

**Description**: Update the VectorDB factory to register the new `SQLiteVecProvider` as the
default. Remove `ChromaDBProvider` from production use (retain file for migration only).

**Tasks**:
- Update `lsm/vectordb/factory.py`:
  - Register `"sqlite"` → `SQLiteVecProvider` as the default
  - Remove `"chromadb"` from the factory's production provider registry
  - Keep the `ChromaDBProvider` class importable for migration code only
- Update `lsm/vectordb/__init__.py` exports
- Update all modules that import from `lsm/vectordb/` to handle the new provider
- Search for all references to `ChromaDBProvider` outside migration code and update them
- Update `lsm/ingest/pipeline.py` to work with the new provider:
  - Replace any ChromaDB-specific calls with `BaseVectorDBProvider` interface calls
  - Ensure the pipeline writes to `lsm_chunks` + `vec_chunks` + `chunks_fts` atomically
- Run: `pytest tests/ -v`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/factory.py` — provider registration
- `lsm/vectordb/__init__.py` — exports
- `lsm/ingest/pipeline.py` — provider integration
- `tests/test_vectordb/test_factory.py` — factory tests
- `tests/test_ingest/test_ingest_pipeline_integration.py` — pipeline integration

**Success criteria**: `create_vectordb("sqlite", config)` returns a `SQLiteVecProvider`.
`create_vectordb("chromadb", config)` raises an error directing users to migrate. The ingest
pipeline writes and reads using the new provider. All existing tests pass or are updated.

---

## 1.4: Phase 1 Code Review and Changelog

**Tasks**:
- Review `SQLiteVecProvider` for SQL injection risks — all queries must use `?` placeholders
- Review FTS5 content-sync triggers — ensure inserts/updates/deletes propagate correctly
- Review transaction boundaries — ensure atomicity across `lsm_chunks` + `vec_chunks` + `chunks_fts`
- Review tests: confirm no mocks or stubs; confirm real sqlite-vec operations
- Verify WAL mode, busy_timeout, and VACUUM behavior
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 1 changes
- Update `docs/user-guide/VECTOR_DATABASES.md`: document SQLite-vec as default, remove
  ChromaDB documentation
- Update `.agents/docs/architecture/packages/lsm.vectordb.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/VECTOR_DATABASES.md`
- `.agents/docs/architecture/packages/lsm.vectordb.md`

**Success criteria**: `pytest tests/ -v` passes. No SQL injection vectors. Changelog and
docs updated.

---
