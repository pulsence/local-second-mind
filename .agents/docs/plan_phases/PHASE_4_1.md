# Phase 4.1: Database Layer Foundation

**Status**: Completed
**Version**: 0.8.1

## Context

`lsm.db` should be the foundational database layer (connections, schema, transactions)
while `lsm.vectordb` should be the functional vector layer that operates on top of it.
Previously the boundary was inverted: `SQLiteVecProvider` owned the shared database connection,
created all 17 application tables (only 3 are vector-specific), and every subsystem reached
into `vectordb.connection` to get DB access. This phase corrects the architecture.

### Problems Addressed

1. **Schema ownership violation** — `sqlite_vec._ensure_schema()` created 14 non-vector
   tables (manifest, memories, schedules, caches, graph, etc.)
2. **Connection inversion** — 5 subsystems extracted connections via
   `getattr(provider, "connection", None)` or `getattr(provider, "_get_conn", None)`
3. **Duplicated resolution logic** — `memory/store.py` and `scheduler.py` each implemented
   identical `_resolve_sqlite_connection()` / `_resolve_postgres_connection_factory()`
4. **Schema duplication** — `lsm_schema_versions` DDL existed in both
   `db/schema_version.py` and `sqlite_vec.py`
5. **Duplicated connection setup** — `stats_cache.py` and `remote/storage.py` created their
   own sqlite connections with different configuration (missing WAL/FK pragmas)

### Design Decisions

- **Keep `PruneCriteria` in `lsm.vectordb.base`** — pruning operates on both `lsm_chunks`
  and `vec_chunks` atomically; it's inherently vector-aware
- **Keep PostgreSQL pool in `postgresql.py`** — pool management is tightly coupled to
  psycopg2/pgvector lazy imports
- **Functional API, no manager class** — matches existing `lsm.db` patterns
- **`lsm_chunks` DDL moves to `lsm.db`** — it's a regular metadata table;
  `vec_chunks` (virtual) stays in vectordb
- **Backward-compatible** — `SQLiteVecProvider.connection` property remains;
  `create_vectordb_provider()` API unchanged

---

## Sub-phases

### 4.1.1: Foundation Modules (Completed)

Created foundational `lsm.db` modules:

- `lsm/db/connection.py` — `create_sqlite_connection()`, `resolve_db_path()`,
  `resolve_sqlite_connection()`, `resolve_postgres_connection_factory()`,
  `resolve_vectordb_provider_name()`
- `lsm/db/schema.py` — `ensure_application_schema()`, `APPLICATION_TABLES` constant
  (14 non-vector tables + 13 indexes)
- `lsm/db/transaction.py` — savepoint-aware `transaction()` context manager with
  module-level `threading.local()` counter
- `lsm/db/__init__.py` — updated exports
- 17 tests in `tests/test_db/`

### 4.1.2: Refactor SQLiteVecProvider Internals (Completed)

- Replaced inline `sqlite3.connect()` + PRAGMA setup with `create_sqlite_connection()`
- Replaced `_resolve_db_path()` with `resolve_db_path()`
- Replaced `_transaction()` method with `transaction()` at all 4 call sites
- Replaced inline schema DDL with `ensure_application_schema()` for non-vector tables
- Replaced hardcoded `health_check` table list with `APPLICATION_TABLES + vector tables`
- Redirected `schema_version._ensure_schema_versions_table()` to `ensure_application_schema()`

### 4.1.3: Consolidate Subsystem Connection Resolution (Completed)

- Replaced 4 private functions in `memory/store.py` with `lsm.db.connection` imports
- Replaced 3 class methods in `scheduler.py` with `lsm.db.connection` imports
- Updated `stats_cache.py` to use `create_sqlite_connection()` for consistent configuration
- Updated `remote/storage.py` to use `create_sqlite_connection()` and `resolve_db_path()`
- Removed duplicated `_resolve_db_path()` from `stats_cache.py` and `remote/storage.py`

### 4.1.4: Code Review and Changelog (Completed)

- Verified no remaining `getattr(provider, "connection")` for connection *resolution*
- Verified no duplicate schema DDL across `lsm.db.schema`, `sqlite_vec.py`, and
  `schema_version.py` (migration.py DDL is for cross-backend targets — different concern)
- Verified `ensure_application_schema` is called exactly once per provider initialization
- Full test suite: 2276 passed, 10 skipped
- Updated `docs/CHANGELOG.md`

## Files Changed

### New
- `lsm/db/connection.py`
- `lsm/db/schema.py`
- `lsm/db/transaction.py`
- `tests/test_db/__init__.py`
- `tests/test_db/test_connection.py`
- `tests/test_db/test_schema.py`
- `tests/test_db/test_transaction.py`

### Modified
- `lsm/db/__init__.py`
- `lsm/db/schema_version.py`
- `lsm/vectordb/sqlite_vec.py`
- `lsm/agents/memory/store.py`
- `lsm/agents/scheduler.py`
- `lsm/ingest/stats_cache.py`
- `lsm/remote/storage.py`
- `docs/CHANGELOG.md`
