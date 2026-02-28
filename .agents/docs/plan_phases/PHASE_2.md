# Phase 2: Agent Data, Manifest, and Sidecar Consolidation

**Status**: Completed

Merges agent memory (`memories.db`), agent schedules (`schedules.json`), the ingest manifest
(`manifest.json`), and runtime sidecars (`stats_cache.json`, remote cache) into the unified
`lsm.db` database. After this phase, `lsm.db` is the single source of truth for all
persistent LSM state.

Reference: [RESEARCH_PLAN.md §3.3, §3.4, §3.9](../RESEARCH_PLAN.md#33-agent-data-consolidation-memoriesdb--schedulesjson)

---

## 2.1: Agent Memory Store — Use Shared lsm.db Connection

**Description**: Modify `SQLiteMemoryStore` to use the shared `lsm.db` connection from
`SQLiteVecProvider` instead of creating its own standalone `memories.db`. The
`lsm_agent_memories` and `lsm_agent_memory_candidates` tables are already created in
Phase 1.

**Tasks**:
- Update `lsm/agents/memory/store.py` (`SQLiteMemoryStore`):
  - Change `__init__` to accept a `connection` (or connection factory) instead of
    creating `sqlite3.connect(sqlite_path)`
  - Remove standalone `memories.db` file creation
  - Update table names from `memories` → `lsm_agent_memories`,
    `memory_candidates` → `lsm_agent_memory_candidates`
  - Update all SQL queries to use the new table names
  - Verify foreign key constraint from `lsm_agent_memory_candidates` →
    `lsm_agent_memories` works correctly
- Remove `MemoryConfig.sqlite_path` from `lsm/config/models/agents.py`
- Update memory store factory/initialization to receive the DB connection from the
  vectordb provider
- Design the connection injection pattern to be backend-agnostic: both
  `SQLiteMemoryStore` and `PostgreSQLMemoryStore` receive a shared connection from
  their respective vectordb provider. The memory store interface should not know
  which backend it runs on.
- Update all tests to use the shared connection pattern

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/agents/memory/store.py` — connection source change, table name changes
- `lsm/config/models/agents.py` — remove `sqlite_path` from `MemoryConfig`
- `lsm/agents/memory/__init__.py` — factory updates
- `tests/test_agents/test_memory_store.py` — updated tests
- `tests/test_agents/test_memory_integration.py` — integration tests with lsm.db

**Success criteria**: `SQLiteMemoryStore` reads/writes to `lsm_agent_memories` in the
shared `lsm.db`. No standalone `memories.db` is created. All memory CRUD operations work.
Foreign key cascades work for memory candidate deletion.

---

## 2.2: Agent Scheduler — Use Shared lsm.db Connection

**Description**: Modify `AgentScheduler` to persist state in the `lsm_agent_schedules` table
in `lsm.db` instead of `schedules.json`.

**Tasks**:
- Update `lsm/agents/scheduler.py`:
  - Change `_save_state_locked()` to INSERT/UPDATE rows in `lsm_agent_schedules`
  - Change `_load_state()` to SELECT from `lsm_agent_schedules`
  - Remove JSON file I/O (`schedules.json` path, `json.dump`/`json.load`)
  - Accept DB connection in `__init__` (same pattern as memory store)
  - Ensure `schedule_id` is deterministic (e.g., `agent_name + schedule_config_hash`)
- Remove `schedules.json` references from `agents_folder` logic
- Update scheduler tests

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/agents/scheduler.py` — DB-backed state
- `tests/test_agents/test_scheduler.py` — updated tests

**Success criteria**: `AgentScheduler` persists all schedule state in `lsm_agent_schedules`.
No `schedules.json` file is created. Schedule state survives process restart via DB.

---

## 2.3: SQLite-Backed Manifest

**Description**: Replace the flat JSON manifest (`manifest.json`) with the `lsm_manifest`
table in `lsm.db`.

**Tasks**:
- Rewrite `lsm/ingest/manifest.py`:
  - `load_manifest()` → SELECT from `lsm_manifest`
  - `save_manifest()` → INSERT/UPDATE `lsm_manifest` rows
  - `get_next_version()` → `SELECT MAX(version) + 1 FROM lsm_manifest`
  - Accept DB connection parameter
  - Retain the same public API signatures for backward compatibility with callers
- Update `lsm/ingest/pipeline.py` to pass the DB connection to manifest functions
- Remove `IngestConfig.manifest` field (already done in 1.1, verify here)
- Update ingest pipeline tests

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/ingest/manifest.py` — DB-backed implementation
- `lsm/ingest/pipeline.py` — connection wiring
- `tests/test_ingest/test_manifest.py` — new tests:
  - Test: manifest tracks file state correctly (mtime, hash, version)
  - Test: get_next_version increments correctly
  - Test: manifest entries persist across connections
  - Test: manifest queries by source_path are indexed
- `tests/test_ingest/test_ingest_pipeline_integration.py` — verify pipeline uses DB manifest

**Success criteria**: No `manifest.json` file is created during ingest. Manifest state lives
in `lsm_manifest` table. Pipeline skip/re-ingest decisions work correctly using DB lookups.

---

## 2.4: Runtime Sidecar Migration

**Description**: Move `stats_cache.json` and remote provider cache files into DB tables.

**Tasks**:
- Move `stats_cache.json` to a `lsm_stats_cache` table (or reuse `lsm_job_status`
  with appropriate metadata):
  - Update `lsm/ingest/stats_cache.py` to read/write from DB
- Move remote provider cache from `remote/*.json` blobs to a `lsm_remote_cache` table:
  - Add `lsm_remote_cache` table to schema (if not already in §3.2):
    ```sql
    CREATE TABLE lsm_remote_cache (
        cache_key TEXT PRIMARY KEY,
        provider TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT
    );
    ```
  - Update `lsm/remote/storage.py` to read/write from DB
- Verify `.lsm_tags.json` files are NOT moved (they are corpus input, not runtime state)
- Update tests for both subsystems

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/sqlite_vec.py` — add `lsm_remote_cache` table to schema creation
- `lsm/ingest/stats_cache.py` — DB-backed stats cache
- `lsm/remote/storage.py` — DB-backed remote cache
- `tests/test_ingest/test_stats_cache.py` — updated tests
- `tests/test_providers/remote/` — updated cache tests

**Success criteria**: No `stats_cache.json` or remote cache JSON blobs are created at
runtime. All cache data lives in `lsm.db`. Cache TTL expiry works via DB queries.

---

## 2.5: Phase 2 Code Review and Changelog

**Tasks**:
- Review agent memory and scheduler for connection lifecycle safety (no leaked connections)
- Review manifest for concurrent access safety (WAL mode handles this, but verify)
- Review sidecar migration for data loss risks
- Review tests: no mocks on DB operations; real SQLite transactions
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `.agents/docs/architecture/development/AGENTS.md` — document DB-backed memory/schedule
- Update `.agents/docs/architecture/packages/lsm.agents.memory.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/AGENTS.md`
- `.agents/docs/architecture/packages/lsm.agents.memory.md`

**Success criteria**: `pytest tests/ -v` passes. No standalone DB or JSON files created
by agent subsystems. Changelog and docs updated.

---
