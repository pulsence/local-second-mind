# v0.8.1 DB Tooling - Research Plan

## Overview

The TODO for v0.8.1 asks for five related pieces of database tooling work:

1. Clarify DB version numbering and storage.
2. Clarify version-to-version DB migration support.
3. Extract migration tooling into an `lsm.migration` package.
4. Add golden data sets for migration testing.
5. Decide whether LSM needs an explicit DB install/init tool.

This document is grounded in the current codebase as inspected on March 5, 2026.
It replaces several stale assumptions from the earlier draft with the current reality of
the repository.

## Research Basis

The following files were reviewed directly:

- `TODO`
- `.agents/docs/RESEARCH_PLAN.md`
- `.agents/docs/plan_phases/PHASE_3.md`
- `.agents/docs/plan_phases/PHASE_4.md`
- `.agents/docs/plan_phases/PHASE_17.md`
- `.agents/docs/plan_phases/PHASE_19.md`
- `lsm/__init__.py`
- `lsm/__main__.py`
- `lsm/db/__init__.py`
- `lsm/db/schema.py`
- `lsm/db/schema_version.py`
- `lsm/db/migration.py`
- `lsm/db/health.py`
- `lsm/db/tables.py`
- `lsm/ui/shell/cli.py`
- `lsm/vectordb/sqlite_vec.py`
- `lsm/vectordb/postgresql.py`
- `tests/test_db/test_health.py`
- `tests/test_vectordb/test_schema_version.py`
- `tests/test_vectordb/test_migration.py`
- `tests/test_vectordb/test_migration_v07.py`
- `tests/conftest.py`

## Executive Findings

The most important research findings are:

1. `lsm_schema_versions` is already a corpus provenance table, not a DB DDL version table.
   It records ingest-time embedding/chunking settings and is referenced by
   `lsm_manifest.schema_version_id`. It should not be repurposed to track DB schema
   migrations.

2. The repository already has substantial DB infrastructure that the old draft treated as
   missing:
   - shared application schema ownership in `lsm/db/schema.py`
   - startup health checks in `lsm/db/health.py`
   - a full explicit `lsm migrate` command in `lsm/db/migration.py`
   - migration resume, validation, auto-detection, legacy v0.7 import, and enrichment

3. The real missing gap is not "how to migrate data between backends" but "how to perform
   in-place DB DDL upgrades on an existing backend". Today, same-backend schema upgrades are
   not a first-class feature.

4. PostgreSQL initialization is no longer missing. `PostgreSQLProvider._ensure_pool()`
   already enables `pgvector` and calls `ensure_application_schema()`. An explicit
   `lsm db init` command would be an operator convenience, not a prerequisite for basic use.

5. The current CLI already exposes `--from-version` and `--to-version` on `lsm migrate`,
   but only the legacy v0.7 path actually uses version information. `--to-version` is
   currently a false affordance and must either be implemented properly or removed.

6. Golden data should move to external archive artifacts, not checked-in fixtures. The
   repo should track the tooling and manifests, while the actual backend snapshots live in
   tar archives resolved through test configuration.

7. Database initialization should become centralized. Once a single bootstrap/init path
   exists, provider constructors should stop auto-creating missing DB state on their own.

8. Embedding-dimension assumptions need normalization as part of this work. The current
   SQLite `FLOAT[384]` schema should be replaced with a dimension derived from the same
   canonical config source used by PostgreSQL initialization and upgrade checks.

9. The existing test layout and fixture conventions differ from the old draft:
   - integration tests live under `tests/test_integration/`
   - reusable fixture artifacts live under `tests/fixtures/` and `tests/test_fixtures/`
   Any golden-data plan should follow those conventions.

10. Runtime/package version strings still report `0.7.1` in `lsm/__init__.py` and
    `pyproject.toml`, even though the phase docs and future planning refer to v0.8.x.
    DB schema tooling must not rely on the package version string alone as the source of
    truth for DB DDL state.

---

## 1. DB Version Numbering

### 1.1 Current State

`lsm/db/schema_version.py` currently records ingest provenance in `lsm_schema_versions`.
The active row is compared against current runtime config using:

- `lsm_version`
- `embedding_model`
- `embedding_dim`
- `chunking_strategy`
- `chunk_size`
- `chunk_overlap`

The table shape in `lsm/db/schema.py` is still:

| Column | Purpose |
|--------|---------|
| `id` | Row identity |
| `manifest_version` | Present but currently always `NULL` |
| `lsm_version` | Application version string |
| `embedding_model` | Corpus embedding model |
| `embedding_dim` | Embedding dimensionality |
| `chunking_strategy` | `structure` or `fixed` |
| `chunk_size` | Chunk size |
| `chunk_overlap` | Chunk overlap |
| `created_at` | Row creation time |
| `last_ingest_at` | Last ingest time |

Important existing semantics:

- `record_schema_version()` inserts a new row when ingest config changes.
- `lsm_manifest.schema_version_id` points at this table.
- `migrate()` copies schema-version rows and may append a newly derived row on the target.
- `check_db_health()` uses this table to detect corpus/config mismatch.

This is already a history of corpus generations, not a singleton record of DB DDL state.

### 1.2 Why the Existing Draft Was Wrong

The earlier draft proposed adding a DB schema integer directly to `lsm_schema_versions`.
That would mix two separate concepts:

- corpus provenance for chunks/files
- database DDL state for tables/indexes/triggers

Those concepts evolve on different schedules and at different cardinalities:

- corpus provenance can have many rows over time
- DB DDL state should be one current state plus an ordered migration log

Repurposing `manifest_version` for DB DDL versioning would have the same problem and would
leave the schema confusing.

### 1.3 Actual Gap

The current codebase has no first-class DB DDL version tracking:

- there is no "current DB schema version" record
- there is no ordered applied-migration log
- startup can detect corpus/config mismatch, but not "your DB DDL is behind the app"
- SQLite has additive column evolution in `_evolve_sqlite_columns()`, but PostgreSQL does not
- there is no way to answer "which DDL migrations have been applied to this database?"

### 1.4 Recommended Design

Keep `lsm_schema_versions` exactly for corpus provenance. Add separate DB DDL state tables.

Recommended storage:

```sql
CREATE TABLE IF NOT EXISTS lsm_db_schema_state (
    singleton_id      INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version    INTEGER NOT NULL,
    last_migration_id TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lsm_db_schema_migrations (
    migration_id      TEXT PRIMARY KEY,
    schema_version    INTEGER NOT NULL,
    description       TEXT,
    applied_at        TEXT NOT NULL,
    applied_by_lsm_version TEXT
);
```

Recommended rules:

1. `lsm_schema_versions` remains the corpus provenance history.
2. `lsm_db_schema_state` is the fast startup check for current DB DDL state.
3. `lsm_db_schema_migrations` is the audit log of applied DDL steps.
4. `manifest_version` should not be repurposed in v0.8.1.
5. `manifest_version` has no future role and should be removed as part of the v0.8.1
   schema cleanup work.
6. Application version is optional metadata only. DB DDL upgrade logic must key off
   `schema_version` and migration IDs, not the app version string.

### 1.5 Version Numbering Policy

Recommended policy:

- Use a monotonic integer `schema_version` for DB DDL generations.
- Use ordered migration IDs such as `0001_initial_schema`, `0002_add_remote_cache_index`.
- Map each release to the highest required `schema_version`, but do not assume one release
  always equals one DB schema version.
- Record the app version that applied a migration only for diagnostics or release notes,
  not as authoritative upgrade state.

This avoids tying DB schema evolution to package-version churn and remains compatible with
the current mismatch between phase/release planning and the repo's current `0.7.1` version
string.

### 1.6 Health/Status Semantics After This Change

After v0.8.1, health checks should distinguish:

- corpus provenance mismatch: current config does not match active `lsm_schema_versions`
- outdated DB schema: DB DDL state is behind the app's required `schema_version`

These should not use the same status code or user guidance.

### 1.7 Corpus Naming Cleanup

Once DB DDL state gets its own tables, `lsm_schema_versions` becomes actively misleading.
The clearer long-term name is `lsm_corpus_versions`, with `lsm_manifest.schema_version_id`
renamed to `corpus_version_id`.

Recommended v0.8.1 cleanup:

1. Rename `lsm_schema_versions` -> `lsm_corpus_versions`.
2. Rename `schema_version_id` -> `corpus_version_id`.
3. Keep `lsm_chunks.version` / `is_current` as the per-file chunk lineage mechanism.
4. If the Python module is renamed to `lsm/db/corpus_version.py`, update all imports in
   the same release instead of leaving a compatibility alias behind.

This keeps the concepts separated:

- corpus generation metadata
- per-file chunk version history
- DB DDL migration state

---

## 2. DB Version Migration Support

### 2.1 Current State

`lsm/db/migration.py` is already a large explicit migration framework. It currently supports:

- Chroma -> SQLite/PostgreSQL backend migration
- SQLite <-> PostgreSQL backend migration
- legacy v0.7 sidecar import
- progress tracking and resume via `lsm_migration_progress`
- row-count validation via `lsm_migration_validation`
- post-migration enrichment
- additive SQLite schema evolution during migration
- backend-specific FTS rebuilds after cross-backend copy

This is materially more capable than the earlier draft assumed.

### 2.2 What It Does Not Support Yet

The missing feature is in-place versioned DB DDL upgrade on an existing backend.

Today there is no public API or CLI for:

- upgrading SQLite vN -> SQLite vN+1 in place
- upgrading PostgreSQL vN -> PostgreSQL vN+1 in place
- applying ordered DDL scripts until a target DB schema version is reached
- previewing such an upgrade with `--dry-run`
- recording which DDL migration IDs were applied

Additional concrete gaps:

1. `_evolve_sqlite_columns()` only handles additive SQLite columns. PostgreSQL has no parallel
   additive evolution path.
2. `ensure_application_schema()` is not a substitute for versioned upgrades. It creates missing
   tables/indexes but does not express ordered migrations.
3. `lsm migrate --to-version` is currently parsed by the CLI but not used by migration logic.
4. Same-backend `migrate("sqlite", "sqlite", ...)` in tests is not a proper in-place upgrade
   story; it is still using the cross-backend copy framework.

### 2.3 Required Conceptual Separation

v0.8.1 needs three distinct workflows:

1. Backend/state migration
   - move data between Chroma, SQLite, PostgreSQL, or legacy v0.7 state
   - should live under `lsm db migrate`

2. DB DDL upgrade
   - apply ordered schema/index/trigger changes in place on the same backend
   - should be a new `lsm db upgrade` workflow

3. Corpus completion/reingest
   - refresh chunks/files because embedding or chunking provenance changed
   - remains `lsm db complete` or ingest-time completion logic

Without that separation the CLI becomes ambiguous and health guidance stays unclear.

### 2.4 Recommended CLI Shape

Recommended command split:

```text
lsm db migrate            # backend/state transfer only
lsm db upgrade            # in-place DB DDL upgrade
lsm db complete           # corpus reprocessing / completion
lsm db check              # health + DDL/version validation
lsm db init               # explicit provisioning convenience
lsm db sync               # one-stop orchestration wrapper
```

Why this is better than overloading the current top-level `lsm migrate` entry point:

- `lsm db` already owns maintenance commands (`prune`, `complete`)
- backend migration and in-place upgrade are different operations
- putting all DB lifecycle commands under `lsm db ...` is more consistent for users
- the current top-level `migrate --to-version` surface is already misleading and should not
  remain half-implemented
- the latest feedback explicitly prefers removing the old top-level `lsm migrate` command
  instead of keeping a temporary alias once `lsm db migrate` exists

Recommended `lsm db sync` behavior:

1. Run centralized DB bootstrap/init if needed.
2. If explicit source arguments are provided, or a legacy/backend source is auto-detected,
   run `lsm db migrate`.
3. Otherwise, if DB DDL state is outdated, run `lsm db upgrade`.
4. If a migration just ran, run `lsm db complete` by default unless `--skip-complete`
   is supplied.
5. If only an upgrade ran, report whether completion is needed but do not auto-run it
   unless `--complete` is explicitly requested.

This matches the user workflow described in the feedback: migration commonly implies a
follow-on completion pass, while a pure DDL upgrade often does not.

### 2.5 Recommended Upgrade Engine

Recommended behavior for `lsm db upgrade`:

1. Open the configured database/backend.
2. Read `lsm_db_schema_state`.
3. Compare it to the app's required `schema_version`.
4. Load all missing migration scripts in order.
5. Apply each script transactionally where the backend allows.
6. Record each applied script in `lsm_db_schema_migrations`.
7. Update `lsm_db_schema_state`.
8. Re-run validation and print an upgrade summary.

Recommended API:

```python
def upgrade_db(
    conn: Any,
    *,
    target_schema_version: int | None = None,
    dry_run: bool = False,
    table_names: TableNames | None = None,
) -> UpgradeResult:
    ...
```

### 2.6 Migration Script Format

Recommended approach:

- keep DDL source-of-truth for base schema in `lsm/db/schema.py`
- add ordered migration modules for changes after the base schema
- each migration module exposes:
  - `migration_id`
  - `schema_version`
  - `description`
  - `apply_sqlite(conn, table_names)`
  - `apply_postgresql(conn, table_names)`

Python modules are a better fit than raw `.sql` files here because:

- the codebase already uses backend-aware Python helpers
- the table prefix is dynamic via `TableNames`
- SQLite/PostgreSQL paths already diverge in places
- migrations may need conditional existence checks

### 2.7 Guarantee Matrix

The TODO asks for three guarantee tiers. Recommended interpretation:

| Tier | v0.8.1 Commitment |
|------|-------------------|
| Full range | Support v0.7.0 -> latest through v0.9.x by chaining legacy import/backend migration + ordered DB upgrades |
| Stepping stone | Support every immediate release-to-release upgrade directly and test it |
| LTR jumps | Design the framework now, but full LTR-to-LTR jump coverage does not become mandatory until v1.0.0 exists |

For v0.8.1 specifically:

- direct v0.7.0 -> current should be a single user command, even if implemented internally as a chain
- adjacent-release upgrade coverage is mandatory
- LTR support should be architectural, not fully populated with fixtures yet

### 2.8 Dry Run and Rollback

Recommended scope:

- `lsm db upgrade --dry-run`: yes
- cross-backend `lsm db migrate --dry-run`: no, not in v0.8.1

Reason:

- dry-running ordered DDL is tractable
- dry-running a full backend copy is expensive and misleading unless it simulates counts,
  target DDL, provider writes, and post-copy rebuilds

Rollback guidance:

- do not promise full automatic rollback for cross-backend migrations in v0.8.1
- for in-place `db upgrade`, rely on backend transactions per migration where possible
- if a migration cannot be transactional, mark it explicitly and add compensating validation
- do not implement downgrade support

### 2.9 Dependency Note

The earlier draft suggested `packaging.version.Version`. That is acceptable only if
`packaging` becomes an explicit runtime dependency. It is not currently listed in the
project dependencies. A simpler alternative is a local semantic-version parser for the
limited release strings LSM uses.

---

## 3. `lsm.migration` Package

### 3.1 Current State

`lsm/db/migration.py` is 1972 lines and currently owns:

- migration source/target enums
- provider resolution
- vector copy
- auxiliary-table copy
- validation bookkeeping
- progress/resume state
- schema evolution
- FTS rebuild hooks
- legacy v0.7 import
- migration auto-detection

Refactoring is justified. The current file is doing too much.

### 3.2 Important Constraint Missing From the Earlier Draft

The old draft proposed moving `lsm/db/enrichment.py` into the migration package. That is
not a good fit with the current codebase.

Today `lsm.db.enrichment` is used by:

- the current `lsm migrate --enrich` flow
- startup health checks via `detect_stale_chunks()`
- standalone enrichment/reporting flows

That makes enrichment broader than migration-only infrastructure. It should stay in
`lsm.db` unless the rest of the DB health/completion API moves with it.

### 3.3 Recommended Package Structure

```text
lsm/migration/
├── __init__.py
├── backend.py        # cross-backend copy logic
├── legacy.py         # v0.7 sidecar import
├── upgrade.py        # in-place DB DDL upgrade engine
├── registry.py       # migration discovery/order metadata
├── progress.py       # progress/resume bookkeeping
├── validation.py     # validation helpers
├── detection.py      # auto-detect migration source
├── types.py          # dataclasses / result models
└── scripts/
    ├── __init__.py
    ├── 0001_initial.py
    └── 0002_...
```

Keep these where they are:

- `lsm/db/schema.py`
- `lsm/db/schema_version.py`
- `lsm/db/enrichment.py`
- `lsm/db/health.py`

### 3.4 Compatibility Strategy

The latest user feedback explicitly rejects compatibility shims for this refactor.

Recommended approach for v0.8.1:

1. Move migration code directly to `lsm.migration`.
2. Update all imports in the codebase, tests, and docs in the same release.
3. Remove stale `lsm.db.migration` imports rather than masking them with aliases.

This is a cleaner cut and avoids carrying module-path debt forward into the new
package structure.

### 3.5 Suggested Refactor Order

Do not start by physically moving everything at once. Recommended order:

1. Introduce `lsm.migration.types`
2. Move pure helpers first (`validation`, `progress`, `detection`)
3. Move legacy/backend flows next
4. Add `upgrade.py`
5. Update remaining imports and delete the old monolith entry point

This reduces merge risk and keeps the migration CLI working throughout the refactor.

---

## 4. Golden Data Sets

### 4.1 Current State

Current migration testing relies on:

- in-memory SQLite connections
- fake providers
- hand-built row seeding
- legacy fixture directories under `tests/test_fixtures/v07_legacy`

That gives good functional coverage, but it is not the same as testing against frozen,
realistic DB states from prior releases.

### 4.2 Where Golden Assets Should Live

The old draft proposed checked-in fixture directories. The new user feedback changes that:

- golden data should be stored as tar archives
- golden data should not be tracked in git

Recommended split:

Tracked in repo:

```text
tests/fixtures/golden/README.md
tests/test_integration/test_golden_migration.py
lsm/migration/golden.py
tests/testing_config.py
```

Stored outside git:

```text
tests/.golden_archives/          # default local cache, gitignored
LSM_TEST_GOLDEN_ARCHIVE_DIR      # override path for CI or shared artifact stores
```

This matches the current test-configuration pattern, which already resolves other test
runtime dependencies from `LSM_TEST_*` environment variables.

### 4.3 Archive Format and Contents

Recommended format: `.tar.gz`

Reason:

- standard tar archive as requested
- gzip compression available through Python stdlib
- easy to unpack in CI and local test runs without new dependencies

Recommended archive layout per supported historical version:

```text
<golden_root>/
└── v0_8_0/
    ├── sqlite.tar.gz
    ├── chroma.tar.gz
    ├── postgresql.tar.gz
    └── SHA256SUMS
```

Each archive should contain:

- backend snapshot or dump
- `golden_manifest.json`
- `config.json`
- corpus/source files or references to the shared corpus bundle
- checksum metadata for internal contents

### 4.4 What Should Be Golden

Recommended golden assets:

| Asset | Recommendation |
|-------|----------------|
| Source corpus | Include in the archive set or a shared corpus archive |
| SQLite DB snapshot | Store as an archive member |
| Chroma snapshot | Store for legacy-coverage versions that still need Chroma migration tests |
| PostgreSQL state | Store as a logical dump archive and support runtime materialization as a fallback |
| Metadata manifest | Include in each archive |
| Config snapshot | Include in each archive |
| Expected row counts | Include in each archive manifest |

### 4.5 PostgreSQL Archive vs Runtime Materialization

The user asked for the differences, so the tradeoff is explicit here:

| Approach | Pros | Cons |
|----------|------|------|
| Store PostgreSQL dump in archive | Restores the real historical PG state directly; fastest path to full backend-to-backend tests; strongest fidelity for PG-specific schema/index behavior | Larger artifacts; dump format can be sensitive to PostgreSQL version differences; needs restore tooling |
| Materialize PostgreSQL state at runtime | Smaller artifact set; avoids dump compatibility issues; easier to regenerate from canonical corpus/SQLite history | Slower test setup; less direct representation of historical PG-native state; can hide PG-specific archival issues |

Recommended decision for v0.8.1:

- store PostgreSQL logical dumps in the external archive set
- keep runtime materialization tooling as a fallback and refresh path

That gives full backend fidelity without depending on git-tracked binary fixtures.
It also avoids coupling golden-data maintenance to old ingest pipelines, which would make
historical fixture generation depend on reconstructing prior application behavior instead of
simply restoring archived DB state.

### 4.6 PostgreSQL Test Isolation Requirement

Golden tests against PostgreSQL must not reuse shared app tables.

Use at least one of:

- unique `table_prefix`
- unique collection name
- isolated test database/schema

Current live PostgreSQL fixtures in `tests/conftest.py` already isolate vector tables by
collection name, but application tables remain shared unless `table_prefix` or database
isolation is added. Golden migration tests should fix that.

### 4.7 Golden Creation Tool

An explicit golden creation tool is useful, but it should follow repo conventions:

Recommended command:

```text
lsm db golden-create --output <dir> [--backend sqlite|postgresql|chroma] [--archive tar.gz]
```

Recommended behavior:

1. validate the source DB state first
2. export `config.json`
3. package the result as a tar archive
4. write `golden_manifest.json` with:
   - backend
   - DB schema version
   - corpus version / provenance record
   - row counts per table
   - collection/table prefix
   - creation timestamp
   - archive checksum
5. emit or copy backend-specific artifacts

### 4.8 Recommended Pragmatic Scope

For v0.8.1, the minimum useful archive set is:

1. one canonical historical corpus archive
2. one Chroma v0.7 archive for the legacy migration window
3. SQLite archives for v0.8.0 and v0.8.1
4. PostgreSQL archives for v0.8.0 and v0.8.1

This satisfies the user's requirement that the golden data exist for Chroma, SQLite, and
PostgreSQL, while keeping the implementation centered on external archive artifacts rather
than git-tracked fixture directories.

### 4.9 Golden Test Matrix

Recommended integration coverage:

- SQLite golden -> current SQLite via `lsm db upgrade`
- SQLite golden -> PostgreSQL via `lsm db migrate`
- PostgreSQL golden/materialized fixture -> SQLite via `lsm db migrate`
- v0.7 legacy fixture -> current SQLite -> `lsm db upgrade`
- validation of row counts, health status, queryability, and FTS rebuilds

Bidirectional SQLite <-> PostgreSQL golden validation is required in v0.8.1 test coverage,
not optional nice-to-have coverage.

---

## 5. DB Install / Init Tool

### 5.1 Current State

The earlier draft said PostgreSQL does not initialize application schema. That is stale.

Current behavior:

- SQLite:
  - `SQLiteVecProvider.__init__()` loads `sqlite-vec`
  - `_ensure_schema()` creates application tables plus vector/FTS tables

- PostgreSQL:
  - `PostgreSQLProvider._ensure_database()` can create the DB and enable `pgvector`
  - `PostgreSQLProvider._ensure_pool()` calls `ensure_application_schema()`
  - vector collection tables are created lazily when embedding dimension is known

- Startup:
  - `lsm/__main__.py` already runs `check_db_health()` for non-`migrate` commands

So an init tool would improve operability and clarity, but it is not filling a total
"no initialization path exists" gap.

### 5.2 Real Value of an Init Tool

An explicit `lsm db init` still has value for:

- CI setup
- pre-provisioning PostgreSQL environments
- surfacing clear, explicit health and version state
- preparing DBs before the first ingest
- creating DB schema without also running a full ingest

### 5.3 Important Backend Nuance

A fully initialized vector backend is not the same thing on SQLite and PostgreSQL.

- SQLite currently creates `vec_chunks` as `FLOAT[384]`, so full vector-table creation is
  possible immediately, but it also exposes an existing 384-dimension assumption.
- PostgreSQL vector collection tables require a concrete embedding dimension. That dimension
  is only known if it can be resolved from config (`global.embedding_dimension`) or an
  explicit CLI flag.

That means `lsm db init` should separate:

1. application schema initialization
2. vector collection/table initialization

### 5.4 Embedding Dimension Normalization

The user feedback explicitly asked to normalize embedding-dimension assumptions. That should
be part of v0.8.1.

Recommended normalization:

1. Treat `global.embedding_dimension` as the canonical dimension used by DB init, upgrade,
   and health checks.
2. If the dimension is absent but the embed model is well-known, resolve it once during
   config validation and store it in config.
3. Change SQLite vector-table DDL to use the resolved dimension instead of hardcoded `384`.
4. Ensure PostgreSQL vector collection initialization uses the same resolved dimension.
5. Add health/check validation that the stored vector-table dimension matches the configured
   or recorded corpus dimension.

This change is a prerequisite for centralized DB initialization. Without it, the init path
cannot create equivalent vector schema across both backends.

### 5.5 Recommended Commands

Recommended additions:

```text
lsm db init
lsm db check
lsm db upgrade
lsm db sync
```

Recommended `lsm db init` behavior:

1. Load config.
2. Ensure backend connectivity.
3. Create database if needed.
4. Enable required extensions (`pgvector` for PostgreSQL).
5. Create application tables via shared schema ownership.
6. If embedding dimension is known, create vector collection schema too.
7. Seed `lsm_db_schema_state` and `lsm_db_schema_migrations` for a fresh DB.
8. Print a clear summary of what was initialized and what remains deferred.

Recommended `lsm db check` behavior:

1. Run `check_db_health()`.
2. Verify DB DDL version state.
3. Verify migration log consistency.
4. Report whether vector collection schema is initialized for the configured dimension.

### 5.6 Centralized Initialization and Auto-Init Policy

The new feedback changes the earlier recommendation: initialization should be centralized,
and it is acceptable for `lsm db init` to run automatically on first use as long as ingest
remains explicit.

Recommended posture:

1. Add a single bootstrap entry point, for example `lsm.db.bootstrap.ensure_db_ready(config)`.
2. Call that bootstrap entry point from CLI/TUI startup and any other central command
   dispatch path.
3. Allow the bootstrap layer to auto-run `lsm db init` on first use.
4. Remove scattered provider-level or module-level "if missing, create schema now" logic
   once the centralized path exists.
5. Keep ingest explicit. Auto-init must not imply an ingest build.

This addresses two goals at once:

- first-run UX stays simple
- DB setup behavior is encapsulated in one place instead of being spread across providers

---

## 6. Recommended Implementation Sequence

Recommended order of work:

1. Add DB DDL state tables and migration registry.
2. Implement `lsm db upgrade`.
3. Add `lsm db check`.
4. Add `lsm db init`.
5. Extract `lsm.migration` package and update all imports directly.
6. Add golden fixture creation tooling.
7. Add golden integration tests.

This order minimizes user-facing confusion:

- version state exists before the upgrade command
- check/init/upgrade semantics are defined before migration-package extraction
- golden tests can target the final CLI/API shape

---

## 7. Impacted Files

Likely implementation touch points:

| Area | Files |
|------|-------|
| New DB DDL state | `lsm/db/schema.py`, `lsm/db/health.py`, new migration registry/scripts |
| In-place upgrade | new `lsm/migration/upgrade.py`, `lsm/__main__.py`, `lsm/ui/shell/cli.py` |
| Migration refactor | `lsm/db/migration.py`, `lsm/db/__init__.py`, new `lsm/migration/*` |
| Init/check commands | `lsm/__main__.py`, `lsm/ui/shell/cli.py`, providers as needed |
| Golden tooling | new `lsm/migration/golden.py` or `lsm/db/golden.py` |
| Golden archives | `tests/.golden_archives/` or `LSM_TEST_GOLDEN_ARCHIVE_DIR`, plus `tests/testing_config.py` |
| Integration tests | `tests/test_integration/test_golden_migration.py` |
| Live PostgreSQL isolation | `tests/conftest.py` and/or PG integration helpers |

---

## 8. Testing Strategy

### 8.1 Unit Tests

- DB DDL state-table creation and reads
- migration registry ordering and duplicate-ID protection
- `db upgrade` dry-run plan output
- applied-migration recording
- `db init` behavior when embedding dimension is known vs unknown
- `db check` status mapping
- `lsm db migrate` CLI cleanup around version flags

### 8.2 Integration Tests

- SQLite historical fixture -> `db upgrade`
- SQLite -> PostgreSQL migration from golden fixture
- PostgreSQL -> SQLite migration from materialized historical fixture
- v0.7 fixture -> current DB chain
- health check before and after upgrade
- app/query smoke test after upgrade

### 8.3 Live PostgreSQL Tests

Mark PostgreSQL golden migration tests appropriately and isolate them with:

- unique database/schema or
- unique `table_prefix` plus cleanup

Do not rely on shared persistent `lsm_` app tables for historical-fixture tests.

Recommended coverage options:

| Coverage level | Pros | Cons |
|----------------|------|------|
| Adjacent-release PostgreSQL coverage only | Lower maintenance, faster live test runs, good coverage for the most likely upgrade path | Weaker confidence for older PG archive restore paths and long-range drift |
| Full historical PostgreSQL coverage | Strongest confidence for supported range; catches archive rot and PG-specific schema drift | Slower and more expensive; more archives to maintain and restore |

Recommended v0.8.1 default:

- require adjacent-release PostgreSQL live coverage
- keep at least one oldest-supported PostgreSQL archive path exercised
- expand toward full historical PG coverage as archived releases accumulate

Recommended test-scope control:

- keep the default repo test target focused on previous-version -> current-version live
  coverage
- add a separate dedicated repo test command/target for the full supported archive and
  migration matrix
- make implementation planning and CI selection target those named commands explicitly

This fits the latest feedback better than an environment-variable switch because the repo
tooling can name and schedule the migration levels directly.

### 8.4 Regression Focus

The most important regressions to guard against are:

1. corrupting corpus provenance history by mixing it with DB DDL versioning
2. silently ignoring `--to-version` semantics
3. upgrading SQLite but not PostgreSQL
4. breaking startup health status meanings
5. failing to remove scattered DB auto-creation paths after central bootstrap is added
6. keeping hardcoded vector dimensions in one backend but not the other

---

## 9. Non-Goals for v0.8.1

These should not be forced into v0.8.1 unless scope changes:

- general schema downgrade support
- full cross-backend dry-run support
- moving all enrichment logic out of `lsm.db`
- compatibility shims for old migration module paths
- git-tracked golden snapshot archives

---

## Clarifications Required

None at this stage. The latest feedback resolved the remaining command-alias and test-target
questions.
