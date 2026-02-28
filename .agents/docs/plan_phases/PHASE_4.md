# Phase 4: Migration System

**Status**: Completed

Implements the `lsm migrate` CLI command for migrating between database backends and
from v0.7.x legacy state. This is the only migration entry point — no automatic migration
at startup or ingest.

Reference: [RESEARCH_PLAN.md §3.7](../RESEARCH_PLAN.md#37-migration-paths)

---

## 4.1: Migration Framework

**Description**: Create the migration framework and CLI entry point.

**Tasks**:
- Create `lsm/db/migration.py`:
  - `MigrationSource` enum: `CHROMA`, `SQLITE`, `POSTGRESQL`, `V07_LEGACY`
  - `MigrationTarget` enum: `SQLITE`, `POSTGRESQL`
  - `migrate(source, target, source_config, target_config, progress_callback)`:
    - Copy vectors, chunk metadata, manifest entries
    - Copy agent memories and schedule state
    - Copy schema versions
    - Re-derive schema versions from active config at migration time
    - Report progress via callback
  - `validate_migration(target_conn)` — verify row counts match
- Add CLI command: `lsm migrate --from <source> --to <target>`
- Add TUI Ingest screen migration warning and "Migrate" action (advisory only)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/migration.py` — migration logic
- CLI entry point — `lsm migrate` command
- `lsm/ui/tui/screens/ingest.py` — migration advisory
- `tests/test_vectordb/test_migration.py`:
  - Test: ChromaDB → SQLite-vec migration preserves all data
  - Test: SQLite-vec → PostgreSQL migration preserves all data
  - Test: PostgreSQL → SQLite-vec migration preserves all data
  - Test: v0.7 legacy state (manifest.json + memories.db + schedules.json) → v0.8 lsm.db
  - Test: migration validation catches row count mismatches
  - Test: migration is idempotent (running twice doesn't duplicate data)

**Success criteria**: `lsm migrate` successfully transfers all state between any supported
source-target pair. Migration never runs automatically.

---

## 4.2: Legacy State Migration (v0.7 → v0.8)

**Description**: Implement the specific v0.7 → v0.8 migration path that imports legacy
`manifest.json`, `memories.db`, and `schedules.json` into the new `lsm.db`.

**Tasks**:
- In `lsm/db/migration.py`, add `_migrate_v07_legacy(source_dir, target_conn)`:
  - Read `manifest.json` → INSERT into `lsm_manifest`
  - Read `memories.db` → INSERT into `lsm_agent_memories` + `lsm_agent_memory_candidates`
  - Read `schedules.json` → INSERT into `lsm_agent_schedules`
  - Read `stats_cache.json` → INSERT into appropriate cache table
  - Read remote cache JSON blobs → INSERT into `lsm_remote_cache`
- Add `lsm migrate --from v0.7 --to v0.8` CLI path
- Create migration integration tests using fixture data

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/migration.py` — v0.7 legacy migration
- `tests/test_vectordb/test_migration_v07.py`:
  - Test: manifest.json import with 100+ entries
  - Test: memories.db import preserves all memory fields
  - Test: schedules.json import preserves all schedule fields
  - Test: missing legacy files handled gracefully (warn, skip)
- `tests/test_fixtures/` — sample legacy state files for testing

**Success criteria**: `lsm migrate --from v0.7 --to v0.8` imports all legacy state into
`lsm.db`. Missing files produce warnings, not errors.

---

## 4.3: Phase 4 Code Review and Changelog

**Tasks**:
- Review migration for data loss risks — verify all fields are preserved
- Review idempotency — verify re-running migration doesn't duplicate
- Review error handling — verify partial failures don't leave corrupt state
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/GETTING_STARTED.md` — add migration instructions for v0.7 users

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/GETTING_STARTED.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---
