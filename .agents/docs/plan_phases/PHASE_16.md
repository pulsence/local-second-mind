# Phase 16: PostgreSQL Parity and Cross-Cutting

**Status**: Pending

Updates the PostgreSQL provider to support all new methods and tables. Implements TUI
startup advisories, transactional consistency verification, and documentation.

Reference: [RESEARCH_PLAN.md §6.1, §6.2, §6.3, §6.8, §6.9](../RESEARCH_PLAN.md#61-tui-startup-advisories-for-offline-jobs)

---

## 16.1: PostgreSQL Provider Updates

**Description**: Update the existing `PostgreSQLProvider` to implement all new methods and
tables that `SQLiteVecProvider` supports.

**Tasks**:
- Update `lsm/vectordb/postgresql.py`:
  - Add all new tables (agent memories, schedules, manifest, reranker cache, graph,
    cluster centroids, job status, embedding models, remote cache) with PostgreSQL-
    appropriate DDL
  - Implement `fts_query()` using PostgreSQL native `tsvector`/`tsquery` and `ts_rank`
  - Implement `prune_old_versions(criteria)` with SQL DELETE
  - Implement `graph_insert_nodes()`, `graph_insert_edges()`, `graph_traverse()` with
    recursive CTEs
  - Add `tsvector GENERATED ALWAYS AS` column on chunks table for native FTS
  - Verify pgvector index works with the same query interface
- Update `PostgreSQLMemoryStore` (if separate from SQLite store) to use shared connection
- Run parity tests: every `test_sqlite_vec.py` test should have a PostgreSQL equivalent

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/postgresql.py` — all new methods
- `tests/test_vectordb/test_postgresql.py` — parity tests
- `tests/test_integration/` — cross-provider integration tests

**Success criteria**: PostgreSQL provider passes all the same interface tests as SQLite-vec.
Migration between providers preserves all data.

---

## 16.2: TUI Startup Advisories

**Description**: On TUI startup, inspect the DB and emit advisory messages for offline
jobs that are configured but not built, or whose output is stale.

**Tasks**:
- Create `lsm/db/job_status.py`:
  - `check_job_advisories(conn, config)` → list of advisory messages
  - Check `lsm_job_status` for:
    - `lsm cluster build` — never run or stale (>20% corpus growth)
    - `lsm graph build-links` — no thematic-link edges
    - `lsm finetune embedding` — no active fine-tuned model
- Update TUI startup (`lsm/ui/tui/app.py` or startup hooks):
  - Display advisories as non-blocking messages
- Also emit advisories after `lsm ingest` on CLI path

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/job_status.py` — advisory logic
- `lsm/ui/tui/app.py` — TUI startup hook
- CLI ingest path — post-ingest advisories
- `tests/test_ui/tui/test_startup_advisories.py`:
  - Test: correct advisory emitted when cluster job never run
  - Test: no advisory when all jobs are current
  - Test: stale detection works (>20% corpus growth)

**Success criteria**: TUI shows actionable advisories on startup. Advisories are non-blocking
and informational.

---

## 16.3: Documentation and Scale Guidance

**Description**: Write user-facing documentation for SQLite scale guidance, privacy model,
and all new features.

**Tasks**:
- Create/update `docs/user-guide/VECTOR_DATABASES.md`:
  - SQLite scale guidance per §6.8 (up to 250k, 250k-1M, >1M)
  - Performance SLO targets
  - Memory estimation formula
  - SQLite deployment defaults (WAL, busy_timeout, VACUUM, ANALYZE)
  - Migration trigger guidance
- Update `docs/user-guide/CONFIGURATION.md`:
  - Complete coverage of all new config fields
  - All removed/renamed fields documented
- Update `docs/user-guide/QUERY_MODES.md`:
  - New ModeConfig structure
  - Retrieval profiles
  - Custom mode creation
- Add privacy labels per §6.9 — which features invoke LLM calls, at ingest vs query time
- Update `example_config.json` with all new fields and defaults
- Update `.env.example` — remove Azure, add any new env vars

- Commit and push changes for this sub-phase.
**Files**:
- `docs/user-guide/VECTOR_DATABASES.md`
- `docs/user-guide/CONFIGURATION.md`
- `docs/user-guide/QUERY_MODES.md`
- `example_config.json`
- `.env.example`

**Success criteria**: Documentation covers all new features, config changes, and breaking
changes. Scale guidance is clear and actionable.

---

## 16.4: Phase 16 Code Review and Changelog

**Tasks**:
- Review PostgreSQL provider parity — every SQLite feature has a PG equivalent
- Review TUI advisory UX — non-blocking, actionable
- Review transactional consistency across providers:
  - Ingest writes to chunk/vector/FTS/manifest tables share one transaction boundary
  - Failure-injection tests prove rollback without partial state
- Review documentation completeness
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`

**Success criteria**: `pytest tests/ -v` passes. PostgreSQL parity verified. Transactional
consistency verified. Documentation complete. Changelog updated.

---
