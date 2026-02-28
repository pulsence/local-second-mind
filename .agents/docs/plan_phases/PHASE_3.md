# Phase 3: Schema Versioning and DB Completion

**Status**: Completed

Implements schema version tracking in the database and incremental corpus update
capabilities. After this phase, LSM can detect when the corpus is stale due to config
changes and selectively re-ingest affected files.

Reference: [RESEARCH_PLAN.md §3.5, §3.6](../RESEARCH_PLAN.md#35-schema-versioning)

---

## 3.1: Schema Version Tracking

**Description**: Implement the `lsm_schema_versions` table logic. Every ingest run records
the embedding model, chunking strategy, and LSM version used. Schema mismatches between
config and the recorded version produce clear errors.

**Tasks**:
- Create `lsm/db/schema_version.py` (or add to existing DB module):
  - `record_schema_version(conn, config)` — INSERT a new row capturing:
    `lsm_version`, `embedding_model`, `embedding_dim`, `chunking_strategy`,
    `chunk_size`, `chunk_overlap`
  - `get_active_schema_version(conn)` → row with the latest `id`
  - `check_schema_compatibility(conn, config)` → returns `(compatible: bool, diff: dict)`
    comparing active version against current config
  - On mismatch: raise `SchemaVersionMismatchError` with clear message listing
    what changed and instructing user to run `lsm migrate`
- Update `lsm/ingest/pipeline.py`:
  - At the start of ingest, call `check_schema_compatibility()`
  - On first ingest (no schema version exists), call `record_schema_version()`
- Update `lsm_manifest` writes to include `schema_version_id` foreign key

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/schema_version.py` — new module (or `lsm/vectordb/schema_version.py`)
- `lsm/ingest/pipeline.py` — schema check at ingest start
- `tests/test_vectordb/test_schema_version.py`:
  - Test: first ingest records schema version
  - Test: matching config passes compatibility check
  - Test: changed `embed_model` fails with clear error
  - Test: changed `chunk_size` fails with clear error
  - Test: manifest entries reference correct schema version

**Success criteria**: Ingest records schema provenance. Config changes that produce
incompatible vectors raise `SchemaVersionMismatchError` with actionable instructions.

---

## 3.2: Versioning Always On and Prune

**Description**: Remove the `enable_versioning` flag. Versioning is the unconditional
operating mode — old chunks are soft-deleted (`is_current=0`) on re-ingest. Add a prune
command for cleanup.

**Tasks**:
- Verify `enable_versioning` removal from `IngestConfig` (done in 1.1)
- Update ingest pipeline to always set `is_current=0` on old chunks during re-ingest
  (never hard-delete during normal ingest)
- Add `prune_old_versions(criteria: PruneCriteria) -> int` to `BaseVectorDBProvider`:
  - `PruneCriteria` dataclass: `max_versions: Optional[int]`, `older_than_days: Optional[int]`
  - Implement in `SQLiteVecProvider`: DELETE from `lsm_chunks`, `vec_chunks`, `chunks_fts`
    WHERE `is_current=0` AND criteria match
  - Returns count of pruned chunks
- Add CLI command: `lsm db prune [--max-versions N] [--older-than-days N]`
- Add `lsm db prune` to the CLI entry point

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/vectordb/base.py` — `prune_old_versions()` abstract method, `PruneCriteria`
- `lsm/vectordb/sqlite_vec.py` — implementation
- `lsm/ingest/pipeline.py` — always soft-delete
- CLI entry point — `lsm db prune` command
- `tests/test_vectordb/test_prune.py`:
  - Test: prune removes only `is_current=0` chunks
  - Test: prune respects `max_versions` criteria
  - Test: prune respects `older_than_days` criteria
  - Test: prune returns correct count

**Success criteria**: Old chunks are never hard-deleted during normal ingest. `lsm db prune`
cleans up soft-deleted chunks based on criteria. Active chunks are never affected.

---

## 3.3: DB Completion — Incremental Corpus Updates

**Description**: Implement selective re-ingest when config changes don't require a full
rebuild. Compare recorded schema version against current config to determine which files
need re-processing.

**Tasks**:
- Create `lsm/db/completion.py`:
  - `detect_completion_mode(conn, config)` → returns the applicable completion mode
    (extension_completion, metadata_enrichment, chunk_boundary_update, embedding_upgrade)
    or `None` if corpus is current
  - `get_stale_files(conn, config, mode)` → list of `source_path` values that need
    re-processing for the detected mode
- Add CLI command: `lsm db complete` — runs selective re-ingest
- Add `--force-reingest-changed-config` flag to `lsm ingest`
- Add `--force-file-pattern <glob>` flag to `lsm ingest` for user-specified selective
  re-ingest of files matching a glob pattern (§3.6.3 "Selective re-ingest" mode)
- Integration with pipeline: when completion mode is detected, pipeline re-ingests only
  affected files instead of skipping them
- Enforce transactional consistency for selective re-ingest:
  - Wrap writes to `lsm_chunks`, `vec_chunks`, `chunks_fts`, and `lsm_manifest` in a
    single transaction boundary
  - On any write failure, rollback the full transaction so no partial state remains

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/completion.py` — completion logic
- `lsm/ingest/pipeline.py` — selective re-ingest integration
- CLI entry point — `lsm db complete` command
- `tests/test_vectordb/test_completion.py`:
  - Test: new file extension in config → only new-extension files re-ingested
  - Test: changed chunk_size → all files re-ingested
  - Test: unchanged config → no files re-ingested
  - Test: metadata enrichment → update metadata without re-embedding
  - Test: injected write failure rolls back chunk and manifest updates atomically

**Success criteria**: `lsm db complete` re-ingests only files whose chunks would differ
based on config changes. Full re-ingest is never triggered unless the embedding model
changes. Failed writes never leave partial chunk/manifest state.

---

## 3.4: Phase 3 Code Review and Changelog

**Tasks**:
- Review schema version comparison logic for edge cases (first run, downgrade, same model
  different version string)
- Review prune safety — verify active chunks cannot be deleted
- Review completion mode detection — verify all four modes trigger correctly
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/CONFIGURATION.md` — document removed fields, new DB commands

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`

**Success criteria**: `pytest tests/ -v` passes. Changelog and docs updated.

---
