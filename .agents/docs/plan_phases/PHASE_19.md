# Phase 19: DB-Agnostic Application Layer

**Status**: Pending

When `db.provider = "postgresql"`, vector operations are available but many application
features are still SQLite-only. This phase removes those backend gaps and makes migration
behavior explicit and testable across SQLite, PostgreSQL, and legacy v0.7 sidecar inputs.

### Cross-Phase DB Test Isolation Requirement (Applies to 19.1-19.7)

All DB-related tests introduced in this phase must be isolation-safe and leave no persistent
artifacts behind.

- Every test that creates tables/indexes/triggers/schemas must clean them up in teardown, or
  run inside an isolated temporary database/schema that is fully dropped after the test.
- SQLite tests must use temporary DB files or in-memory DBs and close all connections after
  completion.
- PostgreSQL tests must use isolated test schemas/databases and explicitly drop them (for
  example via `DROP SCHEMA ... CASCADE`) after completion.
- Connection/cursor lifecycle must be closed or returned to pools in teardown paths, including
  failure paths.
- Test fixtures should include teardown verification that no known test-created tables/schemas
  remain after the suite completes.

---

## 19.1: DB Compatibility Primitives and Connection Resolution

**Description**: Create a reusable SQL compatibility layer and a single connection-resolution
path so DB modules stop reimplementing dialect logic.

**Tasks**:
- Create `lsm/db/compat.py` with shared helpers currently duplicated in `migration.py` and
  `job_status.py`:
  - `dialect(conn)`, `is_sqlite(conn)`, `is_postgres(conn)`
  - `convert_placeholders(query, conn)` for `?` vs `%s`
  - `execute(conn, query, params=())`, `executemany(conn, query, params_seq)`
  - `execute_ddl_script(conn, sql)` — splits multi-statement DDL and executes individually,
    replacing SQLite-only `executescript()` usage (PostgreSQL has no equivalent)
  - `fetchone(conn, query, params=())`, `fetchall(conn, query, params=())`
  - `commit(conn)`, `table_exists(conn, table_name)`
  - `row_to_dict(row, columns)` for tuple/`sqlite3.Row`/mapping rows
  - `insert_returning_id(conn, query, params=())` for SQLite `lastrowid` and PostgreSQL `RETURNING`
  - `safe_identifier(value)` — SQL identifier validation (extract from migration.py `_safe_ident`)
  - `upsert_rows(conn, table, pk, rows)` — dialect-aware UPSERT (extract from migration.py)
  - `count_rows(conn, table_name)` — row counting helper
  - `db_error(*exc_types)` — context manager or helper that catches both `sqlite3.OperationalError`
    and `psycopg2.Error` subtypes, re-raising as a common `DBOperationalError` to normalize
    exception handling across backends
- Ensure `compat.py` keeps PostgreSQL dependencies optional:
  - no unconditional `psycopg2` import at module import time
  - sqlite-only environments must continue to work without PostgreSQL extras installed
- Replace migration-internal duplicates (`_dialect`, `_execute`, `_commit`, `_safe_ident`,
  `_fetch_table_rows`, `_fetch_query_rows`, `_upsert_rows`, `_table_exists`, `_count_table_rows`,
  and row conversion helpers) with `lsm.db.compat` imports.
- Replace `job_status.py`-internal duplicates (`_ph`, `_fetchone`) with `lsm.db.compat` imports.
- Add a unified SQL resolver/context manager in `lsm/db/connection.py`:
  - accepts provider instances and config objects
  - yields SQLite `.connection` or PostgreSQL `._get_conn()` connection
  - documents ownership semantics: SQLite yields a persistent reference (caller must not close),
    PostgreSQL yields a pool-borrowed connection (returned on context exit)
  - documents autocommit differences: sqlite3 uses implicit transactions by default,
    psycopg2 uses explicit transaction blocks — resolver should establish consistent semantics
    for callers (e.g., always autocommit=off, caller uses `compat.commit()`)
  - raises clear errors when provider does not expose SQL access
- Replace ad-hoc `getattr(provider, "connection", ...)` / `getattr(provider, "_get_conn", ...)`
  usage in DB/CLI modules with the new resolver.
- Export compat/resolver API via `lsm/db/__init__.py` if needed by external callers.
- Add tests in `tests/test_db/test_compat.py`:
  - placeholder conversion for both dialects
  - table existence checks for SQLite and PostgreSQL query styles
  - `insert_returning_id` behavior
  - row normalization behavior
  - `execute_ddl_script` splits and runs multi-statement DDL
  - `safe_identifier` accepts valid identifiers, rejects injection attempts
  - `db_error` normalizes exceptions from both backends
  - sqlite-only runtime import test (compat module imports without psycopg2 installed)
- Extend `tests/test_db/test_connection.py` for resolver coverage:
  - SQLite provider instance
  - PostgreSQL provider instance
  - invalid provider error path
  - ownership and commit semantics for both backends
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/compat.py` (new)
- `lsm/db/connection.py`
- `lsm/db/migration.py`
- `lsm/db/job_status.py`
- `lsm/db/__init__.py`
- `tests/test_db/test_compat.py` (new)
- `tests/test_db/test_connection.py`

**Success criteria**: Shared SQL compatibility logic exists in one place, migration and DB
modules consume it, connection resolution no longer relies on ad-hoc provider introspection,
and exception handling is normalized across backends.

---

## 19.2: Dual-Dialect Schema Ownership

**Description**: Make `lsm/db/schema.py` the canonical source of application-table DDL for
both SQLite and PostgreSQL, then remove provider-local schema drift.

**Tasks**:
- Convert `ensure_application_schema()` to backend-agnostic execution:
  - support SQLite and PostgreSQL connections
  - replace `executescript()` (SQLite-only) with `compat.execute_ddl_script()` so DDL
    statements are split and executed individually on PostgreSQL
  - keep `TableNames` as the only table-name source
  - ensure all 14 application tables are covered (explicit list for verification):
    1. `chunks` — main chunk store
    2. `schema_versions` — schema version tracking
    3. `manifest` — ingest manifest
    4. `reranker_cache` — cross-encoder score cache
    5. `agent_memories` — agent long-term memory
    6. `agent_memory_candidates` — agent memory candidates
    7. `agent_schedules` — agent schedule persistence
    8. `cluster_centroids` — clustering centroid storage
    9. `graph_nodes` — knowledge graph nodes
    10. `graph_edges` — knowledge graph edges
    11. `embedding_models` — fine-tuned model registry
    12. `job_status` — offline job status tracking
    13. `stats_cache` — collection stats cache
    14. `remote_cache` — remote provider cache
- Address the chunks table structural divergence between backends:
  - SQLite uses the `chunks` table from schema.py with 24+ normalized metadata columns
  - PostgreSQL provider creates separate `chunks_{collection}` tables with JSONB metadata
  - Decide and document whether PostgreSQL will adopt normalized columns, schema.py will
    emit different DDL per dialect for the chunks table, or the divergence is preserved
    with explicit rationale
- Normalize dialect-specific type usage:
  - integer columns (`INTEGER` vs `BIGINT`)
  - floating point columns (`REAL` vs `DOUBLE PRECISION`)
  - binary columns (`BLOB` vs `BYTEA`) where applicable
  - auto-increment primary keys (`INTEGER PRIMARY KEY AUTOINCREMENT` vs
    `SERIAL PRIMARY KEY` or `GENERATED ALWAYS AS IDENTITY`) — affects `schema_versions`
    and `graph_edges` tables
- Ensure all required indexes are created idempotently on both backends.
- Ensure PostgreSQL provider initialization calls shared `ensure_application_schema()`
  before DB-backed app features run.
- Remove or reduce duplicate provider DDL that overlaps shared schema ownership
  (`_ensure_graph_tables`, `_ensure_embedding_models_table`, and related call paths).
- Keep scheduler persistence schema aligned with shared table definitions.
- Add/expand schema tests:
  - `tests/test_db/test_schema.py` for SQLite idempotency/table/index coverage
  - PostgreSQL-path schema assertions in `tests/test_vectordb/test_postgresql.py`
    (verify all 14 tables are created with correct column types and indexes)
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/schema.py`
- `lsm/db/tables.py` (table name reference — verify APPLICATION_TABLES matches the 14-table list)
- `lsm/vectordb/postgresql.py`
- `lsm/agents/scheduler.py`
- `tests/test_db/test_schema.py`
- `tests/test_vectordb/test_postgresql.py`

**Success criteria**: Shared schema DDL is the single source of truth and creates equivalent
application tables/indexes for both providers without provider-local duplication drift. The
chunks table architecture is explicitly decided and documented.

---

## 19.3: Migration Completeness (PG <-> SQLite + Legacy v0.7 Sidecars)

**Description**: Expand migration scope and tests so cross-backend migration and v0.7 legacy
imports are deterministic, complete, and auditable.

**Tasks**:
- Define and document a migration matrix for every application table (all 14 from 19.2 +
  the vector table + FTS index + migration bookkeeping tables):
  - migrated as data (specify per-table)
  - rebuilt post-migration (specify per-table)
  - intentionally excluded (with rationale per-table)
- Explicitly define handling for migration bookkeeping tables:
  - `migration_progress` and `migration_validation` are target-local operational tables
  - they are not copied from source and must be recreated/reset on target
- Update migration table-copy specs and stage naming so migrated tables are explicit and resume-safe.
- Ensure validation bookkeeping (`lsm_migration_validation`) records all migrated table counts.
- Handle embedding format conversion between backends:
  - SQLite vec0 stores embeddings as binary BLOBs (packed floats via `struct.pack`)
  - PostgreSQL pgvector uses the `vector` type (Python list of floats)
  - Migration must correctly convert between formats in both directions
  - Round-trip tests must verify embedding data survives format conversion within tolerance
- Handle FTS rebuild after cross-backend migration:
  - After PostgreSQL -> SQLite migration, FTS5 virtual table (`chunks_fts`) and its
    sync triggers need rebuilding since they don't exist on PostgreSQL
  - After SQLite -> PostgreSQL migration, the `tsvector` generated column and GIN index
    need creation
- Verify schema evolution logic handles both dialects safely after migration.
- Preserve and harden full v0.7 sidecar import coverage:
  - `manifest.json`
  - `memories.db` (legacy names and prefixed names)
  - `schedules.json`
  - `stats_cache.json`
  - `Downloads/**/*.json` and `remote/**/*.json` caches
- Keep legacy file resolution precedence (`root -> .ingest -> Agents`) and test it.
- Update CLI/help text to reflect full legacy input set (not only manifest/memories/schedules).
- Expand tests in `tests/test_vectordb/test_migration_v07.py`:
  - manifest mapping/defaults
  - memory and memory-candidate imports
  - schedule list and mapping forms, fallback schedule IDs
  - stats cache envelope/map variants
  - remote cache import and cache-key derivation (`query:*`, `feed:rss:*`, `legacy:*`)
  - missing sidecar files are warning-only and non-fatal
- Expand tests in `tests/test_vectordb/test_migration.py`:
  - SQLite -> PostgreSQL -> SQLite round-trip checks
  - PostgreSQL -> SQLite -> PostgreSQL round-trip checks
  - validation mismatch failure behavior remains intact
  - per-table round-trip assertions: row counts, data equality, metadata preservation,
    embedding similarity within floating-point tolerance, and index existence verification
  - FTS index rebuild verification after cross-backend migration
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/migration.py`
- `lsm/db/schema.py`
- `lsm/ui/shell/cli.py`
- `lsm/__main__.py`
- `tests/test_vectordb/test_migration.py`
- `tests/test_vectordb/test_migration_v07.py`

**Success criteria**: Migration behavior is table-explicit, v0.7 sidecar handling is complete,
embedding format conversion is correct in both directions, FTS indexes are rebuilt after
cross-backend migration, and regression tests cover both round-trip backend migration and
legacy sidecar imports with concrete per-table assertions.

---

## 19.4: Convert Remaining SQLite-Only `lsm/db` Modules

**Description**: Remove SQLite-only assumptions from the core DB package so PostgreSQL cannot
silently lose application functionality.

**Tasks**:
- Convert `lsm/db/schema_version.py`:
  - accept backend-agnostic connections
  - use compat execution helpers and backend-safe row handling
  - replace `isinstance(row, sqlite3.Row)` checks with `compat.row_to_dict()`
  - preserve existing mismatch error semantics
- Convert `lsm/db/health.py`:
  - replace SQLite-only table existence checks (`sqlite_master` queries) with
    `compat.table_exists()`
  - support partial-migration checks on PostgreSQL paths
  - keep status model and blocking semantics unchanged
- Convert `lsm/db/completion.py`:
  - remove SQLite-only typing/exception assumptions
  - replace `sqlite3.OperationalError` catches with `compat.db_error()` pattern
  - replace positional row indexing (`row[0]`, `row[1]`) with dict-based access
  - use compat query execution and placeholder handling
  - preserve completion mode decisions
- Convert `lsm/db/job_status.py`:
  - replace local `_ph()` and `_fetchone()` helpers with `lsm.db.compat` imports
  - replace positional row indexing with dict-based access where applicable
- Convert `lsm/db/transaction.py`:
  - remove `sqlite3.Connection` type annotation — accept backend-agnostic connections
  - replace `conn.in_transaction` (SQLite-specific attribute) with backend-safe check
    (PostgreSQL uses `conn.status` or `conn.info.transaction_status`)
  - SAVEPOINT/RELEASE syntax works on both backends, verify semantics match
  - if PostgreSQL transaction model is fundamentally incompatible, document the
    limitation and guard with `compat.is_sqlite()` check
- Refactor `lsm/db/clustering.py`:
  - remove direct `vec_chunks` dependency and `struct.unpack` binary blob parsing
    (SQLite vec0-specific — PostgreSQL pgvector returns Python float lists)
  - add provider-facing embedding retrieval API in `lsm/vectordb/base.py`:
    - `get_embeddings(filters=None, only_current=True) -> Tuple[List[str], List[List[float]]]`
      — batch retrieve chunk IDs and embedding vectors
    - `update_cluster_assignments(updates: List[Tuple[str, int]]) -> None`
      — batch update cluster_id on chunks
  - implement `get_embeddings()` and `update_cluster_assignments()` in both
    `sqlite_vec.py` (reading from vec0 with blob deserialization) and
    `postgresql.py` (reading from pgvector column)
  - address centroid storage on PostgreSQL — `cluster_centroids` table is only created
    by schema.py for SQLite; ensure it is also created on PostgreSQL (19.2 covers DDL,
    but clustering read/write must use compat execution)
  - support centroid read/write on both backends using compat helpers
- Convert `lsm/db/enrichment.py`:
  - remove SQLite-only table checks (`sqlite_master` queries) and placeholder assumptions
  - make PostgreSQL behavior explicit (full support or explicit actionable failure)
  - avoid silent no-op on PostgreSQL
- Add/expand module tests:
  - `tests/test_vectordb/test_schema_version.py` — add PostgreSQL-path tests
  - `tests/test_db/test_health.py` — add PostgreSQL table existence and partial-migration tests
  - `tests/test_vectordb/test_completion.py` — add PostgreSQL-path tests
  - `tests/test_db/test_enrichment.py` — add PostgreSQL-path tests
  - `tests/test_db/test_job_status.py` — verify compat import adoption, no behavior change
  - `tests/test_db/test_transaction.py` — add PostgreSQL-path transaction/savepoint tests
  - clustering tests for both provider paths:
    - `get_embeddings()` returns correct data from both providers
    - centroid read/write works on both backends
    - cluster assignment updates propagate correctly
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/schema_version.py`
- `lsm/db/health.py`
- `lsm/db/completion.py`
- `lsm/db/job_status.py`
- `lsm/db/transaction.py`
- `lsm/db/clustering.py`
- `lsm/db/enrichment.py`
- `lsm/vectordb/base.py`
- `lsm/vectordb/sqlite_vec.py`
- `lsm/vectordb/postgresql.py`
- `tests/test_db/`
- `tests/test_vectordb/`

**Success criteria**: Core DB modules execute with backend-aware behavior on both SQLite and
PostgreSQL, do not silently disable processing paths on PostgreSQL, and no module retains its
own dialect-detection or placeholder helpers (all use `lsm.db.compat`).

---

## 19.5: Consumer Module Parity and Embedding Model Visibility

**Description**: Convert remaining non-DB modules and CLI flows that still rely on SQLite-only
connection assumptions.

**Tasks**:
- Convert ingest/query/cache consumers:
  - `lsm/ingest/pipeline.py` (remove SQLite gate for manifest/schema tracking)
  - `lsm/ingest/api.py` (replace `isinstance(connection, sqlite3.Connection)` check,
    replace `getattr(provider, "connection", None)` with resolver, make `_build_stats_cache`
    backend-agnostic instead of branching on `config.db.provider == "sqlite"`)
  - `lsm/ingest/manifest.py` (backend-agnostic CRUD/upsert path)
  - `lsm/ingest/stats_cache.py` (DB cache path parity)
  - `lsm/query/stages/cross_encoder.py` (replace `INSERT OR REPLACE` and SQLite `datetime('now')`
    with `compat.upsert_rows()` and `CURRENT_TIMESTAMP`/backend-appropriate function)
  - `lsm/query/stages/graph_expansion.py` (replace `getattr(db, "connection", ...)`
    with connection resolver)
  - `lsm/remote/storage.py` (backend-agnostic DB cache storage path)
- Convert `lsm/finetune/registry.py` to backend-agnostic SQL and upsert semantics
  (replace `INSERT OR REPLACE` with `compat.upsert_rows()`, replace `?` placeholders).
- Convert `lsm/finetune/embedding.py`:
  - replace positional row indexing (`row[0]`, `row[1]`, `row[2]`) with dict-based access
  - use compat execution helpers for chunk table queries
- Review agent modules for compat layer adoption:
  - `lsm/agents/memory/store.py` — `SQLiteMemoryStore` uses `executescript()`, `PRAGMA`,
    `sqlite3.Row`, `sqlite3.IntegrityError`; while `PostgreSQLMemoryStore` exists separately,
    both stores should use `lsm.db.compat` where applicable to reduce duplicated dialect logic
  - `lsm/agents/scheduler.py` — already has dual-backend support but implements its own
    helper patterns (`_ensure_schedule_state_schema()` for SQLite, separate PG path);
    adopt compat imports for placeholder conversion, DDL execution, and row handling
- Update finetune CLI commands to remove SQLite-only restrictions for:
  - listing models
  - activating models
  - model registry updates after training
- Update remaining raw SQL call sites in `lsm/ui/shell/cli.py` (cache/graph/clustering/finetune
  helpers) to use resolver + compat patterns.
- Update TUI startup advisory DB access to use resolver/compat instead of ad-hoc provider
  attribute checks:
  - `lsm/ui/tui/app.py` currently branches on `.connection` and `._get_conn()` directly
  - replace with shared resolver path used by CLI/DB modules
- Add/expand tests:
  - `tests/test_ingest/test_manifest.py`
  - `tests/test_ingest/test_stats_cache.py`
  - `tests/test_ingest/test_api.py` (test `_build_stats_cache` backend branching)
  - `tests/test_query/test_cross_encoder.py`
  - `tests/test_query/test_graph_expansion.py`
  - `tests/test_providers/remote/test_storage.py`
  - finetune registry/CLI tests for backend parity (`tests/test_finetune/`)
  - `tests/test_ui/shell/test_cli.py` for updated command behavior
  - `tests/test_ui/tui/test_app.py` startup advisory DB-access parity checks
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/ingest/pipeline.py`
- `lsm/ingest/api.py`
- `lsm/ingest/manifest.py`
- `lsm/ingest/stats_cache.py`
- `lsm/query/stages/cross_encoder.py`
- `lsm/query/stages/graph_expansion.py`
- `lsm/remote/storage.py`
- `lsm/finetune/registry.py`
- `lsm/finetune/embedding.py`
- `lsm/agents/memory/store.py`
- `lsm/agents/scheduler.py`
- `lsm/ui/shell/cli.py`
- `lsm/ui/tui/app.py`
- `tests/test_ingest/`
- `tests/test_query/`
- `tests/test_providers/remote/`
- `tests/test_finetune/`
- `tests/test_ui/shell/test_cli.py`
- `tests/test_ui/tui/test_app.py`

**Success criteria**: Consumer modules and CLI features operate with equivalent behavior on
SQLite and PostgreSQL, including embedding model registry visibility and activation flows.
No consumer module retains ad-hoc `getattr` connection access or SQLite-specific SQL syntax
in CLI or TUI code paths.


---

## 19.6: Debug Phase

User-reported issues and bugs encountered during 19.1–19.5 implementation are resolved
here. The user will provide example output in `<GLOBAL_FOLDER>/Debug/` as needed.

**Tasks**:
- Reproduce each issue from `<GLOBAL_FOLDER>/Debug/` artifacts and capture root cause.
- Implement fixes with explicit regression tests for every reproduced issue.
- Re-run affected sub-phase test suites before closure.
- Ensure all DB tests in this sub-phase satisfy the cross-phase cleanup/isolation requirement.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- Files identified by debug artifacts and associated regression tests

**Success criteria**: Reported debug issues are reproducible, fixed, regression-tested, and
verified in follow-up runs.

---

## 19.7: Code Review and Changelog

**Description**: Final phase-level review and documentation pass to ensure Phase 19 exits with
no unresolved parity or migration gaps.

**Tasks**:
- Review all 19.1–19.6 changes for completeness and unintended regressions.
- Review for deprecated code, dead code, compatibility shims, and circular import risks:
  - verify `lsm/db/compat.py` does not create circular imports — it will be imported by
    many modules (migration, schema, health, completion, clustering, enrichment, job_status,
    transaction, and all consumer modules); ensure the dependency graph is acyclic
  - verify no module retains its own `_dialect`, `_ph`, `_fetchone`, or `_execute` helpers
  - verify no module retains ad-hoc `getattr(provider, "connection", ...)` patterns
- Performance validation:
  - the compat layer adds function call overhead on every SQL operation; verify no measurable
    regression in hot paths (ingest pipeline, query retrieval) via profiling or timing
  - if overhead is significant, consider inlining critical paths or using module-level
    function references
- Validate autocommit/transaction semantics:
  - verify consistent behavior across both backends after conversion — sqlite3 implicit
    transactions vs psycopg2 explicit transaction blocks must produce equivalent commit
    visibility for all consumer modules
- Validate test quality:
  - no auto-pass tests
  - no inappropriate DB mocks for integration behavior
  - coverage includes SQLite and PostgreSQL paths for every converted module
  - coverage includes full v0.7 sidecar artifact set
  - DB tests explicitly verify teardown cleanup and leave no leftover test tables/schemas
    after run completion
  - verify test coverage metrics have not decreased from pre-Phase-19 baseline
- Run full suite: `pytest tests/ -v`.
- Run live PostgreSQL integration subsets where available:
  - `tests/test_vectordb/test_live_postgresql.py`
  - `tests/test_agents/test_live_memory_store_postgresql.py`
- Run targeted migration sanity checks:
  - legacy v0.7 import path
  - PG <-> SQLite migration path
  - embedding format round-trip integrity (verify embeddings survive SQLite blob <-> pgvector
    conversion within floating-point tolerance)
  - FTS index rebuild verification after cross-backend migration
- Update documentation:
  - `.agents/docs/architecture/` where DB architecture changed
  - user-facing migration/backend docs
  - `example_config.json` and `.env.example` if options changed
- Update `docs/CHANGELOG.md` with a complete Phase 19 entry.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- All files modified in 19.1–19.6
- `.agents/docs/architecture/`
- `docs/CHANGELOG.md`
- `example_config.json`
- `.env.example`

**Success criteria**: Phase 19 changes are fully reviewed, tested, documented, and released
with explicit changelog coverage, no unresolved parity blockers, no circular imports, and
no performance regressions in critical paths.

---

*End of Phase 19.*
