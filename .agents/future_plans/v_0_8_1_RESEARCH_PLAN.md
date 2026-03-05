# v0.8.1 DB Tooling — Research Plan

## Overview

v0.8.1 focuses on maturing the database infrastructure introduced in v0.8.0. The work
falls into five areas:

1. **DB version numbering** — clarify how schema versions are defined, tracked, and stored.
2. **DB version migration support** — define migration path guarantees and version ranges.
3. **`lsm.migration` package** — extract migration tooling into a dedicated package.
4. **Golden data sets** — create snapshot fixtures for integration testing across DB backends.
5. **DB installation tool** — replace ad-hoc schema creation with a deliberate init command.

---

## 1. DB Version Numbering

### 1.1 Current State

Schema version tracking lives in `lsm/db/schema_version.py`. The `lsm_schema_versions`
table stores a row per ingest-time configuration snapshot with 10 columns:

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER PK | Auto-increment row ID |
| `manifest_version` | INTEGER | **Always NULL** — unused placeholder |
| `lsm_version` | TEXT | Application version string (e.g. `"0.8.0"`) |
| `embedding_model` | TEXT | Model name used for embeddings |
| `embedding_dim` | INTEGER | Embedding dimensionality |
| `chunking_strategy` | TEXT | `"structure"` or `"fixed"` |
| `chunk_size` | INTEGER | Token/character chunk size |
| `chunk_overlap` | INTEGER | Overlap between chunks |
| `created_at` | TEXT | ISO-8601 timestamp |
| `last_ingest_at` | TEXT | ISO-8601 timestamp |

Compatibility is checked by comparing 6 fields (`SCHEMA_COMPARISON_FIELDS`) between
the active row and the current config. A mismatch raises `SchemaVersionMismatchError`.

### 1.2 Issues to Address

1. **No explicit schema version number.** The system tracks *ingest configuration* but
   not *schema DDL version*. If a table gains a new column or index in v0.8.1, nothing
   in `lsm_schema_versions` records that the DDL changed. The `lsm_version` field records
   the application version, but that is not the same as the schema version.

2. **`manifest_version` is dead weight.** The column is never populated. Its intended use
   should be clarified or the column removed.

3. **No DDL change log.** There is no mechanism to record which DDL changes have been
   applied (analogous to Alembic revision tracking). Schema evolution in `migration.py`
   (`_evolve_schema()`) applies column additions, but there is no audit trail of what
   has already been applied.

4. **Version comparison is configuration-centric, not schema-centric.** Changing the
   embedding model triggers a "version mismatch" even though the schema DDL hasn't changed.
   Conversely, adding a column to `lsm_chunks` would not trigger any compatibility warning.

### 1.3 Design Considerations

**Option A — Semantic schema version integer:**
Add a monotonically increasing `schema_version` integer to `lsm_schema_versions`. Each DDL
change increments this number. The application hard-codes the "current schema version" and
the migration system uses it to determine which DDL changes to apply.

- Pros: Simple, unambiguous, easy to compare.
- Cons: Requires coordination when multiple DDL changes land in the same release.

**Option B — Migration-stamp tracking (Alembic-style):**
Create a `lsm_schema_migrations` table that records applied DDL migration IDs (e.g.
`"0008_add_heading_path_column"`). Each migration is a named, ordered script.

- Pros: Fine-grained, supports branching, standard pattern.
- Cons: More infrastructure, naming conventions, ordering rules.

**Option C — Hybrid approach:**
Keep the semantic version integer for coarse compatibility checks. Use a migrations table
for fine-grained DDL tracking. The integer maps to "all migrations up to this point have
been applied."

- Pros: Best of both — simple version comparisons plus detailed audit trail.
- Cons: Two tables to maintain.

**Recommendation:** Option C (hybrid). The semantic version integer is cheap and enables
fast checks at startup. The migration stamps provide the audit trail needed for incremental
schema evolution and debugging.

### 1.4 Proposed Schema Changes

```sql
-- New table: tracks individual DDL migrations
CREATE TABLE IF NOT EXISTS lsm_schema_migrations (
    migration_id   TEXT PRIMARY KEY,      -- e.g. "0001_initial_schema"
    applied_at     TEXT NOT NULL,          -- ISO-8601
    lsm_version    TEXT NOT NULL,          -- app version that applied it
    description    TEXT                    -- human-readable summary
);

-- Modify lsm_schema_versions:
--   - Add: schema_version INTEGER NOT NULL DEFAULT 1
--   - Decide: keep or drop manifest_version
```

### 1.5 Where Version is Stored and Checked

- **Stored:** In the target database itself (both SQLite and PostgreSQL).
- **Checked:** At startup (via `check_db_health()`) and before ingest (existing
  `check_schema_compatibility()` call in `lsm/ingest/pipeline.py`).
- **Updated:** By the migration system after applying DDL changes, and by the init
  tool when creating a fresh database.

---

## 2. DB Version Migration Support

### 2.1 Current State

The migration system in `lsm/db/migration.py` (~1600 lines) supports:

- **Cross-backend migration:** ChromaDB → SQLite/PG, SQLite ↔ PostgreSQL
- **Legacy migration:** v0.7 sidecar files → v0.8 `lsm.db`
- **Resume:** Stages recorded in `lsm_migration_progress` for crash recovery
- **Validation:** Row-count verification via `lsm_migration_validation`
- **Enrichment:** Post-migration chunk enrichment pipeline (3 tiers)

### 2.2 Migration Path Requirements from TODO

The TODO specifies three migration guarantee tiers:

| Tier | Path | Example |
|------|------|---------|
| **Full range** | v0.7.0 → latest (until v0.9.0) | v0.7.0 → v0.8.1 |
| **Stepping stone** | Immediate previous → immediate following | v0.8.0 → v0.8.1 |
| **LTR jumps** | LTR version → next LTR version | v1.0.0 → v2.0.0 |

### 2.3 Issues to Address

1. **No version-to-version migration concept.** The current system migrates between
   *backends* (SQLite ↔ PG) and from *legacy formats* (v0.7). It does not handle
   *same-backend schema upgrades* (e.g. v0.8.0 SQLite → v0.8.1 SQLite with new columns).

2. **Schema evolution is implicit.** `_evolve_schema()` in migration.py handles adding
   columns, but it runs only during cross-backend migration. There is no path for
   in-place schema upgrades on the same backend.

3. **No version ordering mechanism.** The system cannot determine whether v0.8.0 < v0.8.1
   or which DDL changes are needed to go from one version to another.

4. **No downgrade support.** Migrations are forward-only. The TODO's "SQLite ↔ PG"
   bidirectional requirement is met (data copy), but schema downgrades are not.

5. **No dry-run mode.** Users cannot preview what a migration will do before executing it.

6. **No rollback mechanism.** If validation fails mid-migration, there is no automatic
   rollback. The resume mechanism allows retrying, but not undoing.

### 2.4 Design Considerations

**In-place schema upgrade path:**

The most critical gap is same-backend version upgrades. When a user updates LSM from
v0.8.0 to v0.8.1, the existing database needs DDL changes applied without copying data
to a new backend.

**Proposed approach:**

1. Each release defines a set of ordered DDL migration scripts (see §1.3).
2. On startup or via `lsm migrate --upgrade`, the system:
   a. Reads the current `schema_version` from the DB.
   b. Compares it to the application's expected schema version.
   c. Applies all missing DDL migrations in order.
   d. Records each migration in `lsm_schema_migrations`.
   e. Updates `schema_version` in `lsm_schema_versions`.

**Multi-step migration (full range support):**

For v0.7.0 → v0.8.1, the system needs to chain:
1. v0.7 legacy import (existing `_migrate_v07_legacy()`)
2. v0.8.0 schema creation (existing `ensure_application_schema()`)
3. v0.8.0 → v0.8.1 DDL migrations (new)

This requires a **migration graph** or **ordered migration chain** that the system walks.

**Version ordering:**

Use Python's `packaging.version.Version` (already a dependency via pip/setuptools) or
a simple `tuple` comparison on `(major, minor, patch)`.

### 2.5 Cross-Backend Migration with Version Awareness

When migrating from SQLite v0.8.0 → PostgreSQL v0.8.1:
1. Copy data from SQLite to PostgreSQL (existing backend migration).
2. Apply v0.8.0 → v0.8.1 DDL changes on PostgreSQL (new version migration).

The migration framework should compose backend migration and version migration as
independent, sequenceable stages.

---

## 3. `lsm.migration` Package

### 3.1 Current State

All migration code lives in `lsm/db/migration.py` (~1600 lines). This single file
handles:
- Backend migration (cross-DB copy)
- Legacy migration (v0.7 import)
- Schema evolution
- Validation
- Progress tracking
- Enrichment orchestration

### 3.2 Issues to Address

1. **Single-file monolith.** 1600+ lines in one file makes navigation and maintenance
   difficult.
2. **Tight coupling to `lsm.db`.** Migration logic is entangled with DB layer internals.
3. **No clear extension point for new migration types.** Adding version-to-version
   migration would further bloat the file.

### 3.3 Proposed Package Structure

```
lsm/migration/
├── __init__.py              # Public API: migrate(), upgrade(), validate()
├── backend.py               # Cross-backend migration (SQLite ↔ PG, Chroma → X)
├── legacy.py                # v0.7 → v0.8 legacy format import
├── schema.py                # DDL migration scripts and application logic
├── version.py               # Version comparison, ordering, migration graph
├── progress.py              # Migration progress tracking and resume
├── validation.py            # Post-migration row-count and integrity checks
├── enrichment.py            # Post-migration chunk enrichment (moved from lsm/db/)
└── scripts/                 # Ordered DDL migration scripts
    ├── __init__.py
    ├── 0001_initial_v080.py # Base schema (v0.8.0)
    └── 0002_v081_updates.py # v0.8.1 DDL changes
```

### 3.4 Migration from Current Code

The refactoring should:
1. Move `lsm/db/migration.py` contents into the new package, split by concern.
2. Move `lsm/db/enrichment.py` into the migration package (enrichment is a migration
   concern, not a general DB concern).
3. Keep `lsm/db/schema.py` and `lsm/db/schema_version.py` in `lsm/db/` — these are
   schema *definition* and *tracking*, not migration.
4. Update all imports across the codebase.
5. Update `lsm/db/__init__.py` to re-export from new locations for a transition period
   (or do a clean break if v0.8.1 is considered breaking).

### 3.5 Public API

```python
# lsm/migration/__init__.py

def migrate(
    source: MigrationSource,
    target: MigrationTarget,
    source_config: Any,
    target_config: Any,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    batch_size: int = 1000,
    resume: bool = False,
    skip_enrich: bool = False,
) -> MigrationResult:
    """Cross-backend data migration."""

def upgrade(
    conn: Connection,
    target_version: Optional[str] = None,
    *,
    dry_run: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> UpgradeResult:
    """In-place schema upgrade to target version (default: current app version)."""

def validate(
    conn: Connection,
    *,
    table_names: Optional[TableNames] = None,
) -> ValidationResult:
    """Validate database integrity and schema version."""

def detect_source(
    global_folder: Path,
    config: Any,
) -> DetectionResult:
    """Auto-detect migration source from filesystem and config."""
```

### 3.6 CLI Integration

Extend the existing `lsm migrate` command:

```
lsm migrate                          # Auto-detect and migrate (existing)
lsm migrate --from sqlite --to pg    # Cross-backend (existing)
lsm migrate --from v0.7              # Legacy import (existing)
lsm migrate --upgrade                # In-place schema upgrade (new)
lsm migrate --upgrade --dry-run      # Preview upgrade steps (new)
lsm migrate --validate               # Check DB integrity (new)
```

---

## 4. Golden Data Sets

### 4.1 Current State

Tests use in-memory SQLite databases with hardcoded test data. There are no persistent
snapshot files representing real-world database states.

Relevant test files:
- `tests/test_vectordb/test_migration.py` — uses `_FakeProvider`, `_seed_aux_tables()`
- `tests/test_vectordb/test_migration_v07.py` — uses hardcoded legacy format data

### 4.2 Purpose

Golden data sets serve as frozen reference databases for integration testing:

1. **Version-to-version migration testing:** Verify that v0.8.0 → v0.8.1 upgrades
   produce correct results on a known dataset.
2. **Cross-backend migration testing:** Verify SQLite ↔ PG data fidelity.
3. **Regression testing:** Catch schema changes that break compatibility.
4. **LTR migration testing:** When v1.0.0 ships, its golden set will be used to verify
   migration from v1.0.0 to all future versions.

### 4.3 What a Golden Data Set Contains

For each supported backend (Chroma, SQLite, PostgreSQL):

| Artifact | Format | Contents |
|----------|--------|----------|
| **Database snapshot** | SQLite: `.db` file; PG: `pg_dump` SQL; Chroma: collection dir | Complete DB state |
| **Schema DDL** | `.sql` file | Table definitions at that version |
| **Version manifest** | `golden_manifest.json` | LSM version, schema version, backend, config hash, creation date, row counts per table |
| **Config snapshot** | `config.json` | The configuration used to create the golden set |
| **Sample files** | Directory of source documents | The original files that were ingested |

### 4.4 Golden Data Creation Tool

```
lsm db golden-create                 # Create golden set from current DB state
  --output <dir>                     # Where to write artifacts
  --include-files                    # Also copy ingested source files
  --backend sqlite|postgresql|chroma # Which backend to snapshot
  --version <label>                  # Version label (default: current lsm_version)
```

**Implementation:**

```python
# lsm/migration/golden.py (or lsm/db/golden.py)

def create_golden_set(
    config: LSMConfig,
    output_dir: Path,
    *,
    include_source_files: bool = False,
    version_label: Optional[str] = None,
) -> GoldenManifest:
    """
    Snapshot the current database state as a golden reference set.

    1. Export schema DDL.
    2. Copy/dump the database file.
    3. Record row counts for all application tables.
    4. Save config and version metadata.
    5. Optionally copy ingested source files.
    """
```

### 4.5 Golden Data Storage Location

```
tests/golden_data/
├── v0.8.0/
│   ├── sqlite/
│   │   ├── lsm.db                  # Complete SQLite database
│   │   ├── schema.sql              # DDL snapshot
│   │   ├── golden_manifest.json    # Metadata and row counts
│   │   └── config.json             # Config used to create
│   ├── postgresql/
│   │   ├── pg_dump.sql             # Full pg_dump
│   │   ├── schema.sql              # DDL snapshot
│   │   ├── golden_manifest.json
│   │   └── config.json
│   └── chroma/
│       ├── collection/             # ChromaDB collection directory
│       ├── golden_manifest.json
│       └── config.json
├── v0.8.1/
│   └── ...
└── source_files/                   # Shared sample documents
    ├── sample.pdf
    ├── sample.docx
    ├── sample.md
    └── sample.txt
```

### 4.6 Golden Data Integration Tests

```python
# tests/integration/test_golden_migration.py

class TestGoldenMigration:
    """
    For each golden data version < current version:
    1. Load the golden database.
    2. Run upgrade migration.
    3. Validate all tables exist with expected row counts.
    4. Validate schema version matches current.
    5. Run a sample query to verify data integrity.
    """

    @pytest.mark.parametrize("golden_version", discover_golden_versions())
    def test_upgrade_from_golden(self, golden_version):
        ...

    @pytest.mark.parametrize("source_backend,target_backend", [
        ("sqlite", "postgresql"),
        ("postgresql", "sqlite"),
        ("chroma", "sqlite"),
        ("chroma", "postgresql"),
    ])
    def test_cross_backend_from_golden(self, source_backend, target_backend):
        ...
```

### 4.7 Golden Data for v0.8.0

The first golden set must be created from the v0.8.0 release state. This requires:

1. A representative set of source documents (mix of PDF, DOCX, MD, TXT).
2. A complete ingest cycle producing chunks, embeddings, manifest entries, etc.
3. Some agent activity (memories, schedules) to populate agent tables.
4. Both SQLite and PostgreSQL snapshots.
5. Optionally a ChromaDB snapshot (for testing Chroma → v0.8+ migration).

**Size considerations:** Golden databases should be small enough to commit to the
repository (< 10 MB per backend) but large enough to exercise all table types and
edge cases.

---

## 5. DB Installation Tool

### 5.1 Current State

Database initialization is **implicit and scattered**:

- **SQLite:** `SQLiteVecProvider.__init__()` calls `_ensure_schema()` which calls
  `ensure_application_schema()`. Tables are created when the provider is first
  instantiated (e.g. during ingest or query).
- **PostgreSQL:** `PostgreSQLProvider._ensure_pool()` auto-creates the database and
  enables pgvector, but does **not** call `ensure_application_schema()`. Application
  tables are only created via migration or if the provider is later upgraded via
  Phase 19.
- **Migration:** `_ensure_aux_tables()` in migration.py creates auxiliary tables on
  the target as part of the migration process.

### 5.2 Issues

1. **No single entry point to initialize a fresh database.** Users who configure
   PostgreSQL must either run migration or ingest before the schema exists.
2. **SQLite ≠ PostgreSQL initialization parity.** SQLite auto-creates everything;
   PostgreSQL does not.
3. **Error-prone first run.** If a user starts the TUI before ingesting, queries
   against a PostgreSQL backend will fail because application tables don't exist.
4. **Defensive `IF NOT EXISTS` everywhere.** Because init is implicit, every module
   defensively creates tables, leading to scattered DDL execution.

### 5.3 Design: `lsm db init` Command

```
lsm db init                          # Initialize DB from current config
  --backend sqlite|postgresql        # Override config's db.provider
  --force                            # Drop and recreate (destructive)
  --check                            # Verify schema only, don't create
```

**Behavior:**

1. Read config to determine backend and connection details.
2. For PostgreSQL: create database if missing, enable pgvector extension.
3. Create all application tables (`ensure_application_schema()`).
4. Create vector tables (provider-specific DDL).
5. Record initial schema version.
6. Record initial DDL migration stamps.
7. Print summary: backend, path/connection, tables created, schema version.

**For `--check` mode:**

1. Connect to existing database.
2. Verify all expected tables exist.
3. Verify schema version matches current application version.
4. Verify all DDL migrations have been applied.
5. Report: OK or list of missing tables/migrations.

### 5.4 Startup Integration

When LSM starts (TUI, web server, or CLI), add a lightweight health check:

```python
# lsm/db/health.py (already exists)

def check_startup_health(config: LSMConfig) -> HealthReport:
    """
    Quick check at startup:
    1. Can we connect to the database?
    2. Do application tables exist?
    3. Is the schema version compatible?

    Returns advisory messages, does NOT auto-create or migrate.
    """
```

If the DB is uninitialized, print an advisory:
```
WARNING: Database not initialized. Run `lsm db init` to create schema.
```

If the schema version is outdated, print:
```
WARNING: Database schema is version 1, expected 2. Run `lsm migrate --upgrade`.
```

### 5.5 Removing Implicit Schema Creation

Once `lsm db init` exists, the implicit `_ensure_schema()` calls in provider
constructors can optionally be retained as a safety net or removed to enforce
explicit initialization. The recommendation is:

- **Keep `IF NOT EXISTS` DDL** as a safety net (no harm if tables already exist).
- **Remove provider-level `ensure_application_schema()` calls** from constructors.
- **Add startup health check** to give clear guidance instead of silently creating.

This is a behavioral change that should be carefully considered — removing implicit
creation means `lsm ingest build` on a fresh system would fail without `lsm db init`.
The alternative is to make `lsm db init` implicit (run it automatically on first
detected use) but log clearly that it happened.

---

## 6. Impact Analysis

### 6.1 Files Modified

| Area | Files | Change Type |
|------|-------|-------------|
| New package | `lsm/migration/` (8+ files) | New |
| Schema versioning | `lsm/db/schema_version.py` | Modified |
| Schema DDL | `lsm/db/schema.py` | Modified |
| Table registry | `lsm/db/tables.py` | Modified (new table names) |
| CLI | `lsm/ui/shell/cli.py` | Modified (new subcommands) |
| Health check | `lsm/db/health.py` | Modified |
| Enrichment | `lsm/db/enrichment.py` → `lsm/migration/enrichment.py` | Moved |
| Migration | `lsm/db/migration.py` → `lsm/migration/` split | Moved/split |
| SQLite provider | `lsm/vectordb/sqlite_vec.py` | Modified (init changes) |
| PG provider | `lsm/vectordb/postgresql.py` | Modified (init changes) |
| Golden tests | `tests/integration/test_golden_migration.py` | New |
| Golden data | `tests/golden_data/` | New |
| Golden tool | `lsm/migration/golden.py` | New |

### 6.2 Backward Compatibility

- v0.8.1 databases should be readable by v0.8.0 code (additive schema changes only).
- v0.8.0 databases must be upgradable to v0.8.1 via `lsm migrate --upgrade`.
- The `lsm.migration` package replaces `lsm.db.migration` — old import paths break.
- If v0.8.1 is a breaking release, clean break is acceptable. If not, re-exports from
  `lsm.db.migration` should be maintained for one cycle.

### 6.3 Dependencies

- `packaging` library for version comparison (already available via pip/setuptools).
- No new external dependencies anticipated.

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Schema version tracking: new `schema_version` integer, migration stamp recording.
- Version comparison and ordering logic.
- DDL migration script application and idempotency.
- Golden manifest creation and reading.
- `lsm db init` on fresh SQLite and PostgreSQL backends.

### 7.2 Integration Tests

- **Golden migration tests:** Parameterized over all golden data versions and backend
  combinations (see §4.6).
- **Round-trip migration:** SQLite v0.8.0 → PG v0.8.1 → SQLite v0.8.1.
- **In-place upgrade:** v0.8.0 SQLite → v0.8.1 SQLite (schema-only upgrade).
- **Full chain:** v0.7.0 legacy → v0.8.0 SQLite → v0.8.1 SQLite.
- **Init + ingest + query:** `lsm db init` → `lsm ingest build` → `lsm query` on both
  backends.

### 7.3 Performance Tests

- Golden data creation time (should complete in < 30s for test datasets).
- Schema upgrade time on databases with 10K+ chunks.
- Startup health check latency (should add < 100ms to startup).

---

## 8. Open Questions

These are questions that emerged during research but are not blockers — they are
included here for completeness and can be resolved during implementation planning.

1. **Schema evolution scripts: Python or SQL?** Python scripts offer conditional logic
   and cross-backend abstraction. Raw SQL files are simpler but require per-backend
   variants.

2. **Should `lsm db init` be implicitly run on first use?** If yes, the UX is seamless
   but "magic." If no, users must explicitly initialize, which is clearer but adds a
   step.

3. **Golden data in git or generated?** Committing golden `.db` files to git is simple
   but adds binary bloat. Generating them from scripts is reproducible but slower and
   more fragile. A hybrid (small committed fixtures + generated large sets) may be ideal.

4. **Should the migration package also absorb `lsm/db/schema.py`?** Schema *definition*
   is conceptually different from schema *migration*, but they are closely related. The
   migration scripts need to reference the DDL.

---

## 9. Clarifications Required

1. **Is v0.8.1 a breaking release?** If so, `lsm.db.migration` import paths can be
   removed cleanly. If not, re-exports must be maintained. This affects the scope of
   the `lsm.migration` package refactoring.

2. **v0.7.0 → latest migration chain:** Should the system support direct v0.7.0 → v0.8.1
   migration (single command), or is v0.7.0 → v0.8.0 → v0.8.1 (two steps) acceptable?
   The TODO says "v0.7.0 → latest until 0.9.0" which implies direct, but the
   implementation could chain internally.

3. **LTR migration scope:** The TODO mentions "from LTR until next LTR." Since v1.0.0 is
   the first LTR, this doesn't apply to v0.8.1 yet. Should the migration framework be
   *designed* with LTR jump support in mind, or is it sufficient to add that capability
   when v1.0.0 is released?

4. **Golden data backends:** Should golden data sets be created for all three backends
   (Chroma, SQLite, PostgreSQL) starting from v0.8.0? Chroma is legacy and only needed
   to test Chroma → SQLite/PG migration. Is it worth maintaining a Chroma golden set
   going forward, or only for the v0.7→v0.8 transition?

5. **`manifest_version` column:** Should this column be repurposed (e.g. to hold the new
   schema version integer), dropped entirely, or left as-is? Repurposing avoids a DDL
   migration but creates semantic confusion.

6. **DB init behavior at startup:** Should `lsm db init` be:
   - **(a)** Always explicit — user must run it manually before first use.
   - **(b)** Implicit on first detected use — auto-run with a clear log message.
   - **(c)** Advisory — detect uninitialized DB at startup, print a warning, but don't
     auto-create.

7. **Migration dry-run scope:** Should `--dry-run` apply to both cross-backend migration
   and in-place upgrades, or only to in-place upgrades? Cross-backend dry-run is
   significantly more complex (would need to simulate the copy without writing).

8. **Downgrade support:** The TODO mentions SQLite ↔ PG migration (bidirectional data
   copy). Should *schema* downgrades also be supported (v0.8.1 → v0.8.0), or is
   downgrade strictly a data portability concern (copy data, user manages schema)?
