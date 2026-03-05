# Phase 20: Moving-Average ETA and Migration Graph Index Cleanup

**Status**: Pending

This phase tightens long-running progress reporting and removes a legacy graph-index creation
shim from the migration enrichment path. The ingest/chunking ETA should use a moving average
instead of a whole-run average, migration position backfill should report the same style of ETA,
position-backfill progress must distinguish successful matches from boundary-drift sentinel
writes, long-running migration writes must flush on explicit durable batch boundaries for both
SQLite and PostgreSQL targets, SQLite WAL growth must be controlled during heavy migration
stages, and the graph-node `source_path` index must be guaranteed when graph tables are created
rather than opportunistically during graph backfill.

---

## 20.1: Shared Moving-Average ETA Infrastructure

**Description**: Standardize ETA calculation around a moving window so long-running ingest and
migration operations stop producing unstable whole-run estimates after slow startup or transient
spikes.

**Tasks**:
- Extract or introduce a reusable moving-average ETA helper that accepts timestamp/progress
  samples and returns formatted values such as `ETA 42s`, `ETA 3m 18s`, or `done`.
- Reuse the helper for existing migration ETA code instead of keeping independent ETA
  implementations in multiple modules.
- Replace the ingest/chunking progress ETA in `lsm/ingest/pipeline.py` with a moving-average
  calculation based on recent file/chunk progress instead of total elapsed runtime.
- Preserve current progress-callback shape and CLI output contract; only the ETA math and
  message content should change.
- Ensure the moving-average window is bounded so it stays responsive without reacting to a
  single noisy batch.
- Add targeted tests for:
  - insufficient samples returns an estimating state
  - stable progress produces deterministic ETA formatting
  - slow startup followed by normal throughput no longer poisons the ETA for the full run
  - ingest progress output still emits through the existing callback/CLI path
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/ingest/pipeline.py`
- `lsm/ingest/progress.py` or a new shared progress helper module
- `lsm/db/migration.py`
- `tests/test_ingest/test_progress.py`
- `tests/test_ingest/test_integration.py`

**Success criteria**: Ingest/chunking ETA is computed from a bounded moving average, migration
and ingest share consistent ETA formatting rules, and tests cover both the helper behavior and
the user-visible progress messages.

---

## 20.2: Position Backfill Progress Accounting and ETA

**Description**: Fix position-backfill progress reporting so it reflects what the code is
actually doing. Today `_backfill_positions()` writes `start_char = -1` / `end_char = -1` for
boundary-drifted chunks but only increments `updated` for successful text matches, which makes
the log appear stalled even while rows are still being modified. The fix is to split successful
matches from attempted/deferred writes and report both alongside a moving-average ETA.

**Tasks**:
- Change `_backfill_positions()` to return structured counters instead of a single integer, at
  minimum:
  - `matched_updates`
  - `sentinel_updates`
  - `files_with_writes`
  - `chunks_examined`
- Update the Tier 2 position backfill loop in `lsm/db/enrichment.py` to track recent progress
  samples and log ETA in the same style as simhash/tag/migration progress, but with concrete
  durable-write accounting.
- Replace the misleading single-count log line with one that makes the distinction explicit,
  for example:
  - `[INFO] Position backfill: 2,285/2,693 files (1,005 matched, 1,280 deferred, 2,285 written) (ETA N)`
- Keep boundary-drift sentinel handling (`start_char = -1`, `end_char = -1`) so future runs
  still skip unmatchable chunks, but count those writes as durable progress instead of hiding
  them behind a flat `updated` metric.
- Keep skipped-file handling intact and ensure file-level progress is emitted only after the
  file's writes are included in the current durable batch.
- Add/extend tests for:
  - structured counters returned from `_backfill_positions()`
  - ETA formatting in the revised position-backfill log line
  - boundary-drift sentinel writes contribute to progress accounting
  - idempotent reruns after successful position enrichment
  - boundary-drift sentinel behavior remains intact
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/enrichment.py`
- `tests/test_db/test_enrichment.py`

**Success criteria**: Position backfill emits moving-average ETA progress lines that distinguish
matched chunks from sentinel/deferred writes, so logs no longer imply a flush stall when durable
writes are still occurring. Existing Tier 2 position semantics remain covered by tests.

---

## 20.3: SQLite Position Backfill Durable Batch Flush and WAL Checkpointing

**Description**: Fix the actual SQLite durability behavior for long-running position backfill.
SQLite targets run in WAL mode, commits occur only every five files, and there is currently no
checkpoint policy during Tier 2 position backfill. That combination can leave a very large
`.db-wal` file and make durable progress difficult to observe during Chroma -> SQLite migration.
The fix is to batch writes explicitly, commit on durable boundaries tied to write volume, and
run safe periodic checkpoints during long SQLite-only backfill stages.

**Tasks**:
- Add a SQLite-only durable-write helper for enrichment stages that batches position-backfill
  writes by chunk/file count and flushes on explicit thresholds instead of only every five files.
- Add a SQLite checkpoint helper in the DB utility layer so enrichment code does not inline raw
  `PRAGMA wal_checkpoint(...)` calls. The helper should support:
  - lightweight periodic checkpoints during steady-state work
  - a final stage-close checkpoint after the last durable commit in the position-backfill stage
  - a no-op path for non-SQLite connections
- Tie the flush/checkpoint thresholds to concrete durable-write volume, not just file count. The
  implementation should define and document thresholds for:
  - files processed per flush
  - rows written per flush
  - checkpoint cadence relative to committed batches
- Emit progress only after the corresponding batch commit completes, and if needed report:
  - files processed in the batch
  - rows durably written in the batch
  - cumulative matched/sentinel counts
  - last checkpoint outcome or WAL-control summary
- Keep the SQLite-specific logic isolated to SQLite targets only. PostgreSQL targets must not
  execute SQLite checkpoint code paths.
- Ensure interruption/resume semantics remain correct when the last committed batch is smaller
  than the currently processed path list.
- Add or extend tests for:
  - updated rows are visible on the expected cadence during SQLite position backfill
  - SQLite WAL growth is bounded or reduced after periodic flush/checkpoint operations
  - migration resume remains correct after interrupted SQLite position-backfill work
  - non-SQLite backends do not take SQLite-specific checkpoint paths
- Update logging so the operator can tell whether the SQLite position-backfill stage is still
  making durable progress rather than only parsing files.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/enrichment.py`
- `lsm/db/migration.py`
- `lsm/db/compat.py` or another DB utility module if checkpoint helpers are centralized
- `tests/test_db/test_enrichment.py`
- `tests/test_vectordb/test_migration.py`

**Success criteria**: During SQLite-target position backfill, durable writes occur on predictable
batch boundaries, WAL growth is actively controlled, progress logs reflect committed work, and
resume remains correct after interruption.

---

## 20.4: Batched Durable Writes for Auxiliary and Legacy Table Copy

**Description**: Fix the other migration stages that currently accumulate work inside a single
large transaction before one final commit. `compat.upsert_rows()` commits only once after all
rows are written, which is acceptable for small tables but risky for large auxiliary/legacy
imports on both SQLite and PostgreSQL targets. The concrete write paths that need to change are
`_copy_aux_table()` and the per-artifact UPSERTs inside `_migrate_v07_legacy()`, which currently
build full row lists and then call a one-shot `compat_upsert_rows(...)`. The fix is to make those
table-copy stages flush in committed batches and report progress only after each committed batch.

**Tasks**:
- Extend `lsm/db/compat.py` with a batched UPSERT helper instead of relying on the current
  all-at-once `upsert_rows()` implementation. The helper should:
  - accept a batch size
  - preserve stable column ordering from the first row
  - commit once per batch
  - optionally invoke an `on_batch_committed(rows_committed, total_committed)` callback
- Keep the existing `upsert_rows()` helper for small callers, but make migration-owned copy paths
  use the new batched helper explicitly so the behavior change is deliberate and scoped.
- Change `_copy_aux_table()` to accept a per-batch progress callback and use the batched helper
  instead of one-shot `compat_upsert_rows(...)`.
- In the main `migrate()` loop, pass the current stage id into `_copy_aux_table()` via a callback
  so each committed batch updates:
  - `lsm_migration_progress.rows_processed`
  - user-visible progress output
  before the stage reaches `_complete_stage(...)`.
- If `_copy_auxiliary_state()` remains in the codebase, either convert it to the same batched
  helper or remove it so migration does not retain two code paths with different durability
  semantics.
- Convert each large legacy-import artifact in `_migrate_v07_legacy()` to batched writes with
  a shared helper such as `_write_legacy_rows(...)`, explicitly covering:
  - manifest rows
  - agent memories
  - agent memory candidates
  - schedules
  - stats cache
  - remote cache
- Replace the current single final legacy summary with:
  - per-artifact committed progress updates during the write
  - a final rolled-up summary once all artifact batches are committed
  so long-running legacy imports expose durable forward progress before the last line.
- Keep legacy ID/key derivation unchanged (`schedule_id`, `cache_key`, etc.); only the write and
  progress cadence should change.
- Keep row ordering deterministic within each artifact so partially committed batches remain safe
  for restart/retry without duplicate logical rows.
- Add/extend tests for:
  - `compat.upsert_rows_batched()` commits per batch and invokes its callback with cumulative
    committed-row counts
  - `_copy_aux_table()` advances committed progress in batches for large inputs
  - `_copy_aux_table()` updates `rows_processed` cumulatively before `_complete_stage(...)`
  - legacy import writes each large artifact in batches on SQLite targets
  - the same batched semantics work on PostgreSQL targets without holding one transaction open
    for the entire table
  - per-batch progress reporting reflects committed rows rather than queued rows
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/compat.py`
- `lsm/db/migration.py`
- `tests/test_vectordb/test_migration.py`
- `tests/test_vectordb/test_migration_v07.py`

**Success criteria**: Auxiliary-table and legacy-import migration stages no longer hold a single
large transaction until stage completion. `rows_processed` and progress output advance per
committed batch for both SQLite and PostgreSQL targets, while legacy key/ID semantics remain
unchanged.

---

## 20.5: Commit-Ordered Stage Progress and Tracker Semantics

**Description**: Fix the ordering bugs in migration progress bookkeeping so stage completion and
checkpoint rows are never recorded ahead of the durable commit they are supposed to represent.
The critical concrete case is `run_enrichment_pipeline()`: `_run_stage()` marks a stage
`completed` via `stage_tracker` before the surrounding `commit(conn)` runs for Tier 2 stages.
This sub-phase makes migration progress, enrichment stage tracking, and committed-batch
callbacks all follow the same commit-first ordering.

**Tasks**:
- Refactor `run_enrichment_pipeline()` so stage completion is emitted only after the stage's
  final durable commit. Concretely:
  - `_run_stage()` should not call `stage_tracker(stage_name, "completed")` before the caller's
    commit boundary
  - Tier 2 stage wrappers (`_run_heading_path()`, `_run_positions()`, `_backfill_graph()`) should
    either commit internally before returning success or return enough metadata for the caller to
    commit first and mark complete second
  - the parent aggregate stages (`enrich_tier1`, `enrich_tier2`) should also be marked completed
    only after their child-stage writes are durably committed
- Introduce a shared migration-progress helper in `lsm/db/migration.py` for the common sequence:
  - committed write batch finishes
  - `rows_processed` is updated
  - user-visible progress is emitted
  This helper should replace ad-hoc ordering in stage callbacks where practical and should be the
  only path used by batched auxiliary/legacy copy once `20.4` is implemented.
- Keep the existing safe vector-copy invariant and lock it in with tests:
  - `target_provider.add_chunks(...)` commits its batch
  - only after that does `_set_stage_rows_processed(...)` advance the checkpoint
  - only after that does the progress callback announce the migrated count
- Apply the same commit-first ordering to newly batched auxiliary/legacy stages from `20.4`, so
  `rows_processed` never runs ahead of the just-committed batch.
- Add explicit migration stage wrappers for the remaining finalization steps that currently only
  emit progress text but do not participate in commit-ordered stage bookkeeping:
  - FTS rebuild
  - validation-count recording / migration validation
  If these remain separate visible migration steps, they should use `_begin_stage(...)` /
  `_complete_stage(...)` with completion recorded only after their final commit.
- Ensure `schema_evolution`, the new finalization stages above, and enrichment stage completion
  markers are all recorded only after their DDL or bookkeeping commit has completed.
- Remove any remaining code paths where a stage can be marked `completed` while its writes are
  still pending in the current transaction.
- Add or extend tests for:
  - Tier 2 enrichment stage tracker does not record `completed` before the commit boundary
  - vector-copy `rows_processed` still advances only after provider batch commit
  - auxiliary/legacy batched copy updates `rows_processed` only after each committed batch
  - schema-evolution / FTS-rebuild / validation stages do not report completion ahead of
    committed work
  - no accidental SQLite-only durability code path on PostgreSQL
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/migration.py`
- `lsm/db/enrichment.py`
- `lsm/db/compat.py`
- `lsm/db/transaction.py` if transaction boundary control needs tightening
- `tests/test_vectordb/test_migration.py`
- `tests/test_db/test_enrichment.py`

**Success criteria**: No migration or enrichment stage records `rows_processed`, `completed`, or
user-visible forward progress ahead of the durable commit it represents. Progress ordering is
consistent across vector copy, auxiliary/legacy batch copy, enrichment, and stage-finalization
paths on both SQLite and PostgreSQL targets.

---

## 20.6: Graph Index Ownership at Schema Creation

**Description**: Remove the legacy runtime index-creation check from graph backfill and make
schema creation the only place responsible for ensuring the graph-node `source_path` index.

**Tasks**:
- Verify the canonical schema creation paths for both SQLite and PostgreSQL create
  `idx_<graph_nodes>_source_path` when graph tables are created.
- Remove the runtime `CREATE INDEX IF NOT EXISTS` fallback from graph backfill once schema
  ownership is guaranteed.
- Ensure migration/setup paths call the shared schema creation before any graph backfill stage
  runs, including legacy v0.7 migration flows.
- Add or update tests to prove:
  - schema creation creates the graph-node `source_path` index
  - graph backfill succeeds when the schema-created index already exists
  - migration no longer depends on a legacy graph-backfill-side index creation shim
- Review existing tests that currently assert `_backfill_graph()` creates the index and replace
  them with schema-ownership assertions.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- `lsm/db/schema.py`
- `lsm/db/enrichment.py`
- `lsm/db/migration.py`
- `tests/test_db/test_schema.py`
- `tests/test_db/test_enrichment.py`
- `tests/test_vectordb/test_migration.py`

**Success criteria**: Graph index creation happens during schema setup, graph backfill no longer
contains legacy DDL safety logic, and regression tests validate the schema-first ownership model.

---

## 20.7: Code Review and Changelog

**Description**: Final phase-level review to ensure ETA changes and schema-ownership cleanup do
not introduce regressions in progress reporting or migration behavior.

**Tasks**:
- Review all 20.1–20.6 changes for correctness, backwards compatibility, and dead code.
- Review the final ETA helper placement and import graph so shared progress utilities do not
  introduce circular dependencies between ingest and DB modules.
- Validate that all ETA call sites use the same formatting and estimating/done semantics.
- Validate the migration durability strategy under realistic long-running conditions for both
  SQLite and PostgreSQL targets and confirm it does not introduce durability, resume, or
  performance regressions.
- Run targeted tests for:
  - `tests/test_ingest/test_progress.py`
  - `tests/test_ingest/test_integration.py`
  - `tests/test_db/test_enrichment.py`
  - `tests/test_db/test_schema.py`
  - `tests/test_vectordb/test_migration.py`
  - `tests/test_vectordb/test_migration_v07.py`
- Run full suite: `pytest tests/ -v`.
- Update `docs/CHANGELOG.md` under the Unreleased section with the ETA/reporting and migration
  schema cleanup summary.
- Commit and push changes for this sub-phase using the format in `../COMMIT_MESSAGE.md`.

**Files**:
- All files modified in 20.1–20.6
- `docs/CHANGELOG.md`

**Success criteria**: ETA behavior is consistent across ingest and migration-related workflows,
all audited migration stages make durable visible progress on both SQLite and PostgreSQL targets,
SQLite WAL growth is controlled, graph index ownership is schema-first, all targeted tests pass,
and the changelog captures the phase outcome.

---

*End of Phase 20.*
