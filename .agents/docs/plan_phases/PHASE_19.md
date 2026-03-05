# Phase 19: DB-Agnostic Application Layer

**Status**: Pending

When `db.provider = "postgresql"`, only vector operations work. All application tables (manifest, schema_versions, chunks metadata, graph, clustering, job_status, stats_cache, etc.) are SQLite-only and silently disabled for PostgreSQL. The requirement: when PG is configured, ALL operations use PG; when SQLite is configured, ALL operations use SQLite. The vector and application DB must be the same at all times.

Currently `lsm/ingest/pipeline.py:419-421` drops non-SQLite connections:
```python
manifest_connection = getattr(provider, "connection", None)
if not isinstance(manifest_connection, sqlite3.Connection):
    manifest_connection = None  # All app features silently disabled
```

This phase makes the entire `lsm/db/` layer DB-agnostic and ensures PostgreSQL gets full application table support.

---

## 19.1: DB Compatibility Layer + Unified Provider Interface

### 19.1.1: Create `lsm/db/compat.py` — DB abstraction helpers

**Tasks:**
- Create `lsm/db/compat.py` with dialect-aware utility functions
- Functions: `dialect(conn)`, `is_sqlite(conn)`, `is_postgres(conn)`, `ph(conn)`, `convert_placeholders(query, conn)`, `execute(conn, query, params)`, `executemany(conn, query, params)`, `executescript(conn, script)`, `fetchone(conn, query, params)`, `fetchall(conn, query, params)`, `commit(conn)`, `table_exists(conn, table_name)`, `insert_returning_id(conn, query, params)`, `autoincrement_pk(conn, col)`, `integer_type(conn)`, `real_type(conn)`, `blob_type(conn)`, `fetchone_dict(conn, query, params)`
- All functions accept `conn: Any`, dispatch via `isinstance(conn, sqlite3.Connection)`
- `execute()` auto-converts `?` → `%s` for PostgreSQL
- `executescript()` splits on `;` for PostgreSQL (psycopg2 has no executescript)
- `table_exists()` uses `sqlite_master` for SQLite, `to_regclass()` for PostgreSQL
- `insert_returning_id()` uses `cursor.lastrowid` for SQLite, `RETURNING id` for PostgreSQL
- Write comprehensive unit tests in `tests/test_db/test_compat.py`
- Export from `lsm/db/__init__.py`

**Files:** `lsm/db/compat.py` (new), `lsm/db/__init__.py`, `tests/test_db/test_compat.py` (new)

**Success criteria:** All compat helpers work for both SQLite (real) and PostgreSQL (mocked) connections. Tests pass.

### 19.1.2: Unified provider connection interface

**Tasks:**
- Add abstract `get_connection()` context manager to `BaseVectorDBProvider`
- SQLiteVecProvider: implement as `yield self._conn`
- PostgreSQLProvider: implement as delegate to `_get_conn()`
- Add `resolve_connection(provider)` helper to `lsm/db/connection.py` that works with either provider type
- Write tests for both providers' `get_connection()`

**Files:** `lsm/vectordb/base.py`, `lsm/vectordb/sqlite_vec.py`, `lsm/vectordb/postgresql.py`, `lsm/db/connection.py`, `tests/test_vectordb/test_postgresql.py`, `tests/test_db/test_connection.py`

**Success criteria:** `with provider.get_connection() as conn:` works uniformly for both SQLite and PostgreSQL providers. Existing tests still pass.

### 19.1.3: Code review — Phase 19.1

**Tasks:**
- Review `compat.py` for completeness and edge cases
- Review backwards compatibility of `get_connection()` addition
- Ensure no dead code or unused imports
- Run full test suite: `pytest tests/ -v`
- Commit and push

---

## 19.2: Dual-Dialect Schema + PostgreSQL Application Tables

### 19.2.1: Convert `lsm/db/schema.py` to dual-dialect DDL

**Tasks:**
- Change `ensure_application_schema(conn: sqlite3.Connection, ...)` to accept `conn: Any`
- Replace `conn.executescript()` with `compat.executescript()`
- Generate dialect-aware DDL for all 14 application tables:
  - `AUTOINCREMENT` → `BIGSERIAL PRIMARY KEY` for PostgreSQL
  - `INTEGER` → `BIGINT` for PostgreSQL
  - `REAL` → `DOUBLE PRECISION` for PostgreSQL
  - `BLOB` → `BYTEA` for PostgreSQL
- Keep existing `get_application_tables()` unchanged (no SQL, just names)
- Add indexes for both dialects (same syntax works for both)
- Update tests in `tests/test_db/test_schema.py` (if exists) or create new tests

**Files:** `lsm/db/schema.py`, tests

**Success criteria:** `ensure_application_schema()` creates all 14 tables correctly on both SQLite and PostgreSQL connections. Existing SQLite tests pass unchanged.

### 19.2.2: PostgreSQLProvider creates application tables

**Tasks:**
- In `PostgreSQLProvider._ensure_pool()`, after pool creation, call `ensure_application_schema(conn)` with a PG connection
- Remove redundant `_ensure_graph_tables()` and `_ensure_embedding_models_table()` from PostgreSQLProvider if they're now covered by `ensure_application_schema()`
- Verify `ensure_application_schema()` is idempotent (uses `CREATE TABLE IF NOT EXISTS`)

**Files:** `lsm/vectordb/postgresql.py`, `tests/test_vectordb/test_postgresql.py`

**Success criteria:** When PostgreSQLProvider initializes, all 14 application tables plus vector tables exist in the same PG database.

### 19.2.3: Migrate `lsm/db/migration.py` to use `compat`

**Tasks:**
- Replace `_dialect()`, `_execute()`, `_commit()` in migration.py with `compat.dialect()`, `compat.execute()`, `compat.commit()`
- Replace `_begin_stage()` dual-path with `compat.insert_returning_id()`
- Remove duplicated helper functions that are now in `compat.py`
- Existing migration tests must pass unchanged

**Files:** `lsm/db/migration.py`, `tests/test_db/test_migration.py`

**Success criteria:** Migration uses centralized compat helpers. All existing migration tests pass.

### 19.2.4: Code review — Phase 19.2

**Tasks:**
- Review schema DDL for correctness against both dialects
- Review PostgreSQLProvider init flow for table creation order
- Verify no duplicate table creation (schema.py vs provider-specific methods)
- Run full test suite
- Commit and push

---

## 19.3: Make `lsm/db/` Modules DB-Agnostic

### 19.3.1: Convert `lsm/db/schema_version.py`

**Tasks:**
- Replace `conn: sqlite3.Connection` → `conn: Any`
- Replace `conn.execute()` → `compat.execute()`
- Replace `cursor.lastrowid` → `compat.insert_returning_id()`
- Replace `sqlite3.Row` check with generic dict/tuple handling
- Update tests

**Files:** `lsm/db/schema_version.py`, `tests/test_db/test_schema_version.py`

**Success criteria:** `check_schema_compatibility()`, `record_schema_version()`, `get_active_schema_version()` work with both connection types.

### 19.3.2: Convert `lsm/db/job_status.py`

**Tasks:**
- Replace inline `_ph()`, `_fetchone()` with `compat.ph()`, `compat.fetchone()`
- Remove `import sqlite3` (only used for isinstance, now in compat)
- Update `record_job_status()` to use `compat.execute()` instead of raw conn.execute/cursor.execute branching
- Verify `ON CONFLICT ... EXCLUDED` upsert works for both (it does — same syntax)
- Update tests

**Files:** `lsm/db/job_status.py`, `tests/test_db/test_job_status.py`

**Success criteria:** All job_status functions work with both connection types using compat helpers.

### 19.3.3: Convert `lsm/db/health.py`

**Tasks:**
- Replace `sqlite_master` queries → `compat.table_exists()`
- Enable full health check for PostgreSQL (currently short-circuits)
- Replace `sqlite3.Connection` type hints → `Any`
- Update schema compatibility check to work with PG connections
- Update tests

**Files:** `lsm/db/health.py`, `tests/test_db/test_health.py`

**Success criteria:** `check_db_health()` works when provider is PostgreSQL, checking PG tables.

### 19.3.4: Convert `lsm/db/completion.py`

**Tasks:**
- Replace `sqlite3.OperationalError` catches with generic `Exception` handling
- Replace `conn: sqlite3.Connection` → `conn: Any`
- Replace `?` placeholders via `compat.execute()`
- Update tests

**Files:** `lsm/db/completion.py`, `tests/test_db/test_completion.py`

**Success criteria:** Completion mode detection works with PostgreSQL connections.

### 19.3.5: Convert `lsm/db/clustering.py`

**Tasks:**
- Add `get_embeddings_for_clustering()` method to `BaseVectorDBProvider` (abstract)
- Implement in `SQLiteVecProvider`: query `vec_chunks` with struct.pack/unpack
- Implement in `PostgreSQLProvider`: query `chunks_*` table, return embeddings as lists
- Update `build_clusters()` and `get_top_clusters()` to accept a provider or connection
- Replace direct `vec_chunks` SQL with provider API call
- Replace `?` placeholders for centroid storage via `compat.execute()`
- Update tests

**Files:** `lsm/db/clustering.py`, `lsm/vectordb/base.py`, `lsm/vectordb/sqlite_vec.py`, `lsm/vectordb/postgresql.py`, tests

**Success criteria:** Clustering works end-to-end with both SQLite and PostgreSQL providers.

### 19.3.6: Convert `lsm/db/enrichment.py`

**Tasks:**
- Replace all 4 `sqlite_master` queries → `compat.table_exists()`
- Replace `conn: sqlite3.Connection` → `conn: Any`
- Replace all `?` placeholders via `compat.execute()`
- Handle `vec_chunks` references: for PostgreSQL, enrichment syncs are done on the PG chunks table directly
- Replace `conn.commit()` → `compat.commit()`
- Update tests

**Files:** `lsm/db/enrichment.py`, `tests/test_db/test_enrichment.py`

**Success criteria:** All enrichment tiers work with PostgreSQL connections.

### 19.3.7: Code review — Phase 19.3

**Tasks:**
- Review all 6 converted modules for correctness
- Verify no remaining `sqlite3.Connection` type hints in `lsm/db/` (except `connection.py` resolvers)
- Verify no remaining `sqlite_master` queries in `lsm/db/`
- Verify no remaining raw `?` placeholders (all go through compat)
- Run full test suite
- Commit and push

---

## 19.4: Convert Consumer Modules Outside `lsm/db/`

### 19.4.1: Convert `lsm/ingest/pipeline.py`

**Tasks:**
- Replace lines 419-421 (`if not isinstance(manifest_connection, sqlite3.Connection): manifest_connection = None`) with `provider.get_connection()` usage
- Replace all raw SQL (manifest writes, graph edges, reranker cache) with `compat.execute()` calls
- Replace `INSERT OR IGNORE` with `compat`-based upsert
- Enable transactional manifest writes for PostgreSQL
- Update tests

**Files:** `lsm/ingest/pipeline.py`, `tests/test_ingest/test_pipeline.py`

**Success criteria:** Full ingest pipeline works with PostgreSQL provider — manifest tracked, schema version recorded, enrichment runs.

### 19.4.2: Convert `lsm/ingest/manifest.py`

**Tasks:**
- Replace `conn: Optional[sqlite3.Connection]` → `conn: Optional[Any]`
- Replace `?` placeholders via `compat.execute()`
- Replace `INSERT ... ON CONFLICT` syntax to work with both dialects (already standard SQL, just needs placeholder conversion)
- Update tests

**Files:** `lsm/ingest/manifest.py`, `tests/test_ingest/test_manifest.py`

**Success criteria:** Manifest CRUD works on PostgreSQL.

### 19.4.3: Convert `lsm/ingest/stats_cache.py`

**Tasks:**
- Replace `sqlite3.Connection` → `Any`
- Replace `?` placeholders via `compat.execute()`
- Replace `INSERT OR REPLACE` with `ON CONFLICT DO UPDATE`
- Update connection creation to support PG via provider
- Update tests

**Files:** `lsm/ingest/stats_cache.py`, `tests/test_ingest/test_stats_cache.py`

**Success criteria:** Stats cache works on PostgreSQL.

### 19.4.4: Convert `lsm/query/stages/cross_encoder.py`

**Tasks:**
- Replace `INSERT OR REPLACE` with `ON CONFLICT DO UPDATE`
- Replace `datetime('now')` with `compat`-aware timestamp
- Replace `?` placeholders via `compat.execute()`
- Update tests

**Files:** `lsm/query/stages/cross_encoder.py`, tests

**Success criteria:** Reranker cache works on PostgreSQL.

### 19.4.5: Convert `lsm/query/stages/graph_expansion.py`

**Tasks:**
- Replace raw `conn.execute()` with `compat.execute()`
- Replace `getattr(db, "connection", ...)` with `provider.get_connection()`
- Update tests

**Files:** `lsm/query/stages/graph_expansion.py`, tests

**Success criteria:** Graph expansion queries work on PostgreSQL.

### 19.4.6: Convert `lsm/remote/storage.py`

**Tasks:**
- Replace `sqlite3.Connection` types → `Any`
- Replace `INSERT OR REPLACE` with `ON CONFLICT DO UPDATE`
- Replace `?` placeholders via `compat.execute()`
- Update `_open_cache_connection()` to support PG connections
- Update tests

**Files:** `lsm/remote/storage.py`, tests

**Success criteria:** Remote cache works on PostgreSQL.

### 19.4.7: Convert `lsm/finetune/registry.py`

**Tasks:**
- Replace `conn: sqlite3.Connection` → `conn: Any`
- Replace `INSERT OR REPLACE` with `ON CONFLICT DO UPDATE`
- Replace `?` placeholders via `compat.execute()`
- Update tests

**Files:** `lsm/finetune/registry.py`, tests

**Success criteria:** Embedding model registry works on PostgreSQL.

### 19.4.8: Convert `lsm/ui/shell/cli.py` raw SQL

**Tasks:**
- Replace raw SQL in cache clear, cluster visualize, graph build commands
- Use `provider.get_connection()` instead of `getattr(provider, "connection", None)`
- Replace `?` placeholders and `INSERT OR IGNORE` with compat calls
- Update tests

**Files:** `lsm/ui/shell/cli.py`, tests

**Success criteria:** All CLI commands work with PostgreSQL provider.

### 19.4.9: Update `lsm/db/connection.py`

**Tasks:**
- Add unified `resolve_connection(provider) -> ContextManager` that works for both provider types
- Deprecate `resolve_sqlite_connection()` (keep for backward compat, add deprecation warning)
- Update tests

**Files:** `lsm/db/connection.py`, tests

**Success criteria:** Single resolver works for both backends.

### 19.4.10: Code review — Phase 19.4

**Tasks:**
- Verify no remaining `import sqlite3` outside `lsm/db/` and `lsm/vectordb/` (except tests)
- Verify no remaining raw SQL outside `lsm/db/` that isn't wrapped by compat
- Verify no remaining `getattr(provider, "connection", None)` patterns (replaced by `get_connection()`)
- Run full test suite
- Commit and push

---

## 19.5: Debug Phase

**Tasks:**
- User will manually test PostgreSQL end-to-end: configure `db.provider = "postgresql"`, run `lsm migrate`, then `lsm ingest`, then `lsm query`
- Debug output will be placed in `.lsm/Debug/`
- Fix any issues discovered during manual testing

---

## 19.6: Final Review

### Code Review

**Tasks:**
- Review all phases and ensure no gaps or bugs
- Review all changes for backwards compatibility
- Review for deprecated code, dead code, or legacy compatibility shims — remove them
- Review all new modules for proper error handling and logging
- Review test suite:
  - No mocks or stubs on database operations (unit tests may use mocks for PG connections)
  - No auto-pass tests
  - Test structure matches project module structure
  - All new features have unit + integration tests
- Review security:
  - All queries use parameterized queries (never string interpolation for values)
  - Table names validated via `TableNames` registry
- Commit and push

### Integration Testing

**Tasks:**
- Run full test suite: `pytest tests/ -v --cov=lsm --cov-report=html`
- Run end-to-end with SQLite provider (regression)
- Run end-to-end with PostgreSQL provider (new capability)

### Architecture Documentation Update

**Tasks:**
- Update `.agents/docs/architecture/` files to reflect DB-agnostic layer
- Update `ARCHITECTURE.md` top-level overview
- Commit and push

**Files:** All architecture docs

**Success criteria:** Architecture docs accurately reflect codebase.

### Changelog

**Tasks:**
- Update `docs/CHANGELOG.md` with Phase 19 changes
- Commit and push

---

*End of Phase 19.*
