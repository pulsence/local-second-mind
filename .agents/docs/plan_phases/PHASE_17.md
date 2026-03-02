# Phase 17: Debug Phase

**Status**: Completed

Pre-release debug phase for resolving implementation gaps and bugs identified during the
Phase 11–16 review, plus any additional issues surfaced by the user.

**Key concern addressed**: Migration must not just copy data between backends — it must
also bring existing chunks up to date with the current ingest pipeline. Phases 3
(versioning + metadata), 7 (heading path, intelligent chunking), 12 (simhash dedup),
14 (multi-vector summaries + node types), and 15 (graph construction) all introduced
per-chunk processing that older chunks may lack. Section 17.7 defines a tiered
post-migration enrichment pipeline to handle this.

---

## 17.1: Issues Identified During Phase 11–16 Review (Historical, Re-Verified)

The following issues were cataloged during a comprehensive review of phases 11–16. Each
item includes severity, affected phase, location, and description.

These entries are historical findings at review time. See the re-verification snapshot
below for the current codebase status as of March 1, 2026.

### Phase 12 Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 12-A | Medium | Simhash not computed during ingest — `simhash` column exists in schema (`lsm/db/schema.py:64`) but `lsm/ingest/pipeline.py` never imports `compute_minhash` or stores the value. Dedup works at query time (on-the-fly), but the pre-computed optimization path is incomplete. | `lsm/ingest/pipeline.py` |
| 12-B | Low | `docs/user-guide/CONFIGURATION.md` missing documentation for Phase 12 config fields: `dedup_threshold`, `mmr_lambda`, `max_per_section`, `temporal_boost_enabled`, `temporal_boost_days`, `temporal_boost_factor`. | `docs/user-guide/CONFIGURATION.md` |

### Phase 13 Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 13-A | Medium | Run summary (`_write_run_summary()`) does not include context-level tracking metadata (context labels seen, per-label iteration counts, per-label conversation/response IDs). Phase 13.2 explicitly required this. | `lsm/agents/harness.py:1119-1175` |
| 13-B | Low | `docs/user-guide/AGENTS.md` and `.agents/docs/architecture/development/AGENTS.md` still show only `query_knowledge_base` — new pipeline tools (`query_context`, `execute_context`, `query_and_synthesize`) not documented. | `docs/user-guide/AGENTS.md`, `.agents/docs/architecture/development/AGENTS.md` |

### Phase 14 Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 14-A | Medium | Optional dependencies `umap-learn` and `hdbscan` not declared in `pyproject.toml` optional-dependencies. Code handles missing HDBSCAN gracefully but installation guidance is absent. | `pyproject.toml` |
| 14-B | Medium | `lsm cluster visualize` CLI command not implemented. Phase 14.2 specifies UMAP HTML plot export. | `lsm/__main__.py` |

### Phase 15 Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 15-A | Medium | `lsm graph build-links` CLI command not implemented. Should build thematic links offline using cosine similarity above threshold between chunk embeddings. | `lsm/__main__.py` |
| 15-B | Low | `finetune_enabled: bool = False` not added to `GlobalConfig`. Phase 16.2 TUI startup advisories depend on this config trigger. | `lsm/config/models/global_config.py` |

### Phase 16 Issues

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 16-A | High | TUI startup advisories broken for PostgreSQL — `_check_startup_advisories()` accesses `.connection` property which doesn't exist on `PostgreSQLProvider` (has `._pool` instead). Silent failure. | `lsm/ui/tui/app.py:352-354` |
| 16-B | High | `job_status.py` `check_job_advisories()` signature accepts `sqlite3.Connection` only — incompatible with PostgreSQL `psycopg2` connections. | `lsm/db/job_status.py:29` |
| 16-C | Major | PostgreSQL parity tests remain incomplete: graph and prune coverage now exist, but FTS parity coverage is still missing. Phase 16.1 success criteria requires PostgreSQL provider to pass the same interface tests as SQLite-vec. | `tests/test_vectordb/test_postgresql.py` |
| 16-D | Medium | CLI post-ingest advisories not implemented. Phase 16.2 specifies "Also emit advisories after `lsm ingest` on CLI path." | `lsm/ui/shell/cli.py`, `lsm/__main__.py` |

### Re-Verification Snapshot (March 1, 2026)

- Resolved in current codebase: `12-A`, `12-B`, `13-A`, `13-B`, `14-A`, `14-B`,
  `15-A`, `15-B`, `16-A`, `16-B`, `16-D`.
- Remaining from this historical set: `16-C` (specifically PostgreSQL FTS parity tests).
- Scope implication for Phase 17 execution: keep 17.1 as historical context, but treat
  `16-C` as the only still-open item from this list unless new regressions appear.

### Remaining Open Item From 17.1

- `16-C` closure task: add PostgreSQL FTS parity tests mirroring SQLite-vec FTS behavior
  checks, and make this a blocking exit criterion for 17.10.

---

## 17.2: Table Name Infrastructure

**Description**: Replace all hardcoded `lsm_` table name strings across the codebase with
a centralized `TableNames` registry. This enables the configurable `table_prefix` introduced
in 17.3 and eliminates scattered string literals.

**Tasks**:
- Create `lsm/db/tables.py` with a `TableNames` dataclass:
  - `prefix: str = "lsm_"` field
  - One `@property` per table returning `f"{self.prefix}<base_name>"` for all 17 tables:
    - Core tables (14): `chunks`, `schema_versions`, `manifest`, `reranker_cache`,
      `agent_memories`, `agent_memory_candidates`, `agent_schedules`,
      `cluster_centroids`, `graph_nodes`, `graph_edges`, `embedding_models`,
      `job_status`, `stats_cache`, `remote_cache`
    - Migration bookkeeping (1): `migration_progress` (created in 17.6)
    - SQLite-only virtual tables (2): `vec_chunks`, `chunks_fts` — these do not
      exist in PostgreSQL (pgvector uses its own extension tables). Properties
      still exist on `TableNames` for use by SQLite provider code only
  - `validate()` method rejecting non-alphanumeric/underscore prefixes
  - `application_tables() -> tuple[str, ...]` returning resolved core table names
    (excluding optional migration bookkeeping tables)
- Update `lsm/db/schema.py`:
  - `ensure_application_schema(conn, table_names: TableNames = None)` — default instance if None
  - `APPLICATION_TABLES` tuple replaced by `get_application_tables(prefix) -> tuple`
  - All DDL strings reference `table_names.X` instead of hardcoded `"lsm_*"` literals
- Propagate `TableNames` parameter into all production files that contain hardcoded
  table name references (currently ~22 files):
  - DB layer: `schema.py`, `migration.py`, `job_status.py`, `schema_version.py`,
    `clustering.py`, `completion.py`
  - Vectordb layer: `sqlite_vec.py`, `postgresql.py`
  - Ingest layer: `pipeline.py`, `manifest.py`, `stats_cache.py`, `graph_builder.py`
  - Agent layer: `memory/store.py`, `scheduler.py`
  - Finetune layer: `registry.py`, `embedding.py`
  - Query layer: `stages/cross_encoder.py`, `stages/graph_expansion.py`
  - Remote layer: `storage.py`
  - UI layer: `shell/cli.py`
  - Main: `__main__.py`
- Update all 15 test files that create tables inline to use `TableNames`:
  `test_sqlite_vec.py`, `test_prune.py`, `test_migration_v07.py`, `test_migration.py`,
  `test_completion.py`, `test_embedding.py`, `test_schema.py`, `test_schema_version.py`,
  `test_job_status.py`, `test_clustering.py`, `test_graph_expansion.py`,
  `test_versioning.py`, `test_stats_cache.py`, `test_scheduler.py`, `test_graph_builder.py`

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/tables.py` — new centralized table name registry
- `lsm/db/schema.py` — parameterized DDL
- All production files listed above (currently ~22) — replace hardcoded table strings
- All 15 test files listed above — use `TableNames` for table setup
- `tests/test_db/test_tables.py` — unit tests for `TableNames`:
  - Test: default prefix `"lsm_"` produces `lsm_chunks`, `lsm_manifest`, etc.
  - Test: custom prefix `"app_"` produces `app_chunks`, `app_manifest`, etc.
  - Test: empty prefix produces `chunks`, `manifest`, etc.
  - Test: invalid prefix with special characters raises `ValueError`
  - Test: `application_tables()` returns correct tuple length and values

**Success criteria**: Zero hardcoded `lsm_` table name strings remain in production code.
All table references flow through `TableNames`. Existing tests pass with default prefix.

---

## 17.3: Config Restructure (`vectordb` → `db`)

**Description**: Rename the JSON config key from `"vectordb"` to `"db"` and restructure
the config model. `DBConfig` owns database-level settings (`table_prefix`, `path`,
`connection_string`, host/port/database/user/password). A nested `VectorConfig` submodel
holds vector-specific settings (`provider`, `collection`, `index_type`, `pool_size`).
Rename the Python attribute from `.vectordb` to `.db` across all files.

**Tasks**:
- Restructure `lsm/config/models/vectordb.py`:
  - Rename `VectorDBConfig` → `VectorConfig` containing only vector-specific fields:
    `provider`, `collection`, `index_type`, `pool_size`
  - Create `DBConfig` dataclass with:
    - `table_prefix: str = "lsm_"`
    - `path: Path = Path("Data")`
    - `connection_string: Optional[str] = None`
    - `host`, `port`, `database`, `user`, `password` (PostgreSQL connection fields)
    - `vector: VectorConfig = field(default_factory=VectorConfig)`
  - `DBConfig` provides delegation properties (`.provider`, `.collection`, etc.) forwarding
    to `self.vector.*`
  - `DBConfig.validate()` validates prefix and calls `self.vector.validate()`
- Update `lsm/config/models/lsm_config.py`:
  - Rename field `vectordb: VectorDBConfig` → `db: DBConfig`
  - Update `__post_init__`: resolve `self.db.path`
  - Update `validate()`: call `self.db.validate()`
  - Update shortcut properties: `persist_dir` → `self.db.path`, `collection` → `self.db.collection`
- Update `lsm/config/models/__init__.py`:
  - Export `DBConfig`, `VectorConfig`
  - Remove `VectorDBConfig`
- Update `lsm/config/loader.py`:
  - Replace `build_vectordb_config()` with `build_db_config(raw)` reading `raw["db"]`
  - Extract `table_prefix`, `path`, connection fields from `db` section
  - Extract `vector` subobject for `VectorConfig`
  - Update `config_to_raw()` to serialize as `"db": {"table_prefix": ..., "path": ..., "vector": {...}}`
- Rename `.vectordb` → `.db` across all ~75 files (~220 occurrences). Clean rename, no
  aliases or shims. Affected areas:
  - Config layer: `lsm_config.py`, `loader.py`, `__init__.py`, `constants.py`
  - DB layer: `schema.py`, `migration.py`, `connection.py`, `job_status.py`, `completion.py`
  - Vectordb layer: `factory.py`, `sqlite_vec.py`, `postgresql.py`, `base.py`
  - Ingest layer: `pipeline.py`, `api.py`, `manifest.py`, `stats.py`, `tagging.py`, `graph_builder.py`
  - Query layer: `pipeline.py`, `context.py`, `retrieval.py`, `planning.py`,
    `stages/dense_recall.py`, `stages/sparse_recall.py`, `stages/graph_expansion.py`,
    `stages/multi_vector.py`
  - Agent layer: `harness.py`, `scheduler.py`, `memory/store.py`, `tools/__init__.py`,
    `tools/similarity_search.py`, `tools/extract_snippets.py`
  - UI layer: `tui/app.py`, `tui/commands/ingest.py`, `tui/screens/ingest.py`,
    `tui/screens/query.py`, `tui/state/settings_view_model.py`,
    `tui/presenters/query/provider_info.py`, `tui/widgets/settings_vectordb.py`,
    `shell/cli.py`, `shell/commands/agents.py`
  - Other: `eval/cli.py`, `remote/chain.py`, `remote/providers/news/rss.py`, `finetune/registry.py`
  - Tests (25+): `conftest.py`, all test fixtures with `"vectordb"` in config dicts,
    `test_loader.py`, `test_models_vectordb.py`, `test_settings_view_model.py`, `test_app.py`,
    `test_migration.py`, `test_factory.py`, `test_postgresql.py`, `test_chromadb.py`, etc.
- Update `example_config.json`:
  ```json
  "db": {
    "table_prefix": "lsm_",
    "path": "Data",
    "vector": {
      "provider": "sqlite",
      "collection": "local_kb"
    }
  }
  ```
- Wire `table_prefix` into `TableNames`: construct `TableNames(prefix=config.db.table_prefix)`
  in provider/config plumbing and pass through to all SQL-using modules

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/vectordb.py` — `VectorConfig` + `DBConfig`
- `lsm/config/models/lsm_config.py` — `.db` field
- `lsm/config/models/__init__.py` — updated exports
- `lsm/config/loader.py` — `build_db_config()`, `config_to_raw()`
- `example_config.json` — new `"db"` structure
- All ~75 files with `.vectordb` references — renamed to `.db`
- `tests/test_config/test_loader.py` — updated config parsing tests
- `tests/test_config/test_models_vectordb.py` — updated model tests:
  - Test: `build_db_config()` parses `"db"` format correctly
  - Test: `DBConfig` delegation properties forward to `VectorConfig`
  - Test: `table_prefix` validation rejects special characters
  - Test: `config_to_raw()` round-trips correctly
  - Test: loading config with old `"vectordb"` key fails with clear error

**Success criteria**: Config loads from `"db"` key. All consumer code uses `config.db.*`.
`table_prefix` flows from config through `TableNames` into all SQL table references.

---

## 17.4: Database Health Check at Startup

**Description**: Add a database health check that runs at both TUI and CLI startup before
any operations. Detects version mismatches, missing/corrupt databases, legacy provider
state, and partially completed migrations. Provides clear guidance on resolution.

**Tasks**:
- Create `lsm/db/health.py` with:
  - `DBHealthReport` dataclass: `status`, `details`, `suggested_action`, `schema_diff`
  - `status` values: `"ok"`, `"missing"`, `"mismatch"`, `"corrupt"`,
    `"partial_migration"`, `"legacy_detected"`, `"stale_chunks"`
  - `check_db_health(config: LSMConfig) -> DBHealthReport` running all checks:
    1. **Database reachable?**
       - SQLite: check if `lsm.db` exists at configured path.
         Missing on first run is OK (status `"missing"`, non-blocking)
       - PostgreSQL: run a short connection check (`SELECT 1`) using configured
         connection settings; unreachable DB is `"corrupt"` with actionable guidance
    2. **Legacy provider detected?** — Check for `.chroma/` directory in `global_folder`.
       Status `"legacy_detected"` with migration instructions
    3. **Schema version compatible?** — Use existing `check_schema_compatibility()` from
       `lsm/db/schema_version.py`. Status `"mismatch"` with field-level diff
    4. **Required tables present?** — Verify all `table_names.application_tables()` exist
       using `_table_exists()`. Status `"corrupt"` if any are missing
    5. **Partial migration?** — Check `table_names.migration_progress` table (from 17.6) for
       rows with `status = 'in_progress'` or `'failed'`. Status `"partial_migration"`.
       **Implementation note**: This check depends on the migration_progress table created
       in 17.6. Implement as a stub that returns `"ok"` if the table does not exist, then
       wire the real logic after 17.6 is complete.
    6. **Stale chunks?** — Reuse `detect_stale_chunks()` from 17.7 for deterministic counts
       (not random spot checks), including `simhash IS NULL`, `node_type IS NULL`,
       `heading_path IS NULL` on headed chunks, and missing section/file summary nodes when
       summaries are enabled. Status `"stale_chunks"` (non-blocking) with suggested action:
       `lsm migrate --enrich` or `lsm ingest --force-reingest-changed-config`.
       **Implementation note**: This check depends on `detect_stale_chunks()` from 17.7.
       Implement as a stub that returns `"ok"` during this sub-phase, then wire the real
       logic after 17.7 is complete.
- Wire into TUI startup (`lsm/ui/tui/app.py`):
  - Call `check_db_health()` in `_check_startup_advisories()` before provider init
  - Display notification for any non-`"ok"` status with the `suggested_action` text
- Wire into CLI startup (`lsm/__main__.py`):
  - Call `check_db_health()` after config load, before dispatching any subcommand
  - Print warning and exit with guidance for blocking statuses (`"mismatch"`,
    `"corrupt"`, `"partial_migration"`, `"legacy_detected"`)
  - `"missing"` is non-blocking (first run)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/db/health.py` — health check module
- `lsm/ui/tui/app.py` — TUI startup integration
- `lsm/__main__.py` — CLI startup integration
- `tests/test_db/test_health.py`:
  - Test: clean database → `"ok"`
  - Test: no database file → `"missing"`
  - Test: PostgreSQL unreachable → `"corrupt"` with connection guidance
  - Test: `.chroma/` directory present → `"legacy_detected"` with migration instructions
  - Test: schema version mismatch → `"mismatch"` with correct field diff
  - Test: missing application tables → `"corrupt"`
  - Test: partial migration in progress → `"partial_migration"`
  - Test: chunks with NULL `simhash` → `"stale_chunks"` (non-blocking)
  - Test: chunks with NULL `node_type` → `"stale_chunks"` (non-blocking)
  - Test: missing section/file summaries when enabled → `"stale_chunks"` (non-blocking)
  - Test: all chunks fully enriched → `"ok"`

**Success criteria**: TUI and CLI detect database state issues on startup and display
actionable guidance before any operations fail. No more cryptic errors from loading
incompatible databases. Stale chunk detection warns users about missing enrichment.

---

## 17.5: Migration CLI Restructure & Auto-Detection

**Description**: Restructure the migration CLI argument parser, fix v0.7 subfolder
resolution, and add one-shot auto-detection. This sub-phase establishes the new CLI
surface for all subsequent migration features.

**Tasks**:
- Update argument parser in `lsm/__main__.py`:
  - Replace `--from`/`--to` with:
    - `--from-db {chroma,sqlite,postgresql}` (optional for one-shot)
    - `--to-db {sqlite,postgresql}` (optional, default: configured provider)
    - `--from-version <version>` (optional for one-shot)
    - `--to-version <version>` (optional, default: current)
    - `--resume` flag (wired in 17.6)
    - `--enrich` flag (wired in 17.8)
    - `--skip-enrich` flag (wired in 17.8)
  - Remove old `--from`/`--to` arguments entirely
  - Legacy compatibility rule: `--from-version v0.7` without `--from-db` enters
    legacy file migration mode (v0.7 does not map cleanly to a DB backend)
- Fix v0.7 subfolder resolution in `lsm/db/migration.py`:
  - Add `_resolve_v07_file(source_dir: Path, filename: str) -> Optional[Path]` that
    searches `source_dir/`, then `source_dir/.ingest/`, then `source_dir/Agents/`
  - Update `_migrate_v07_legacy()` to use `_resolve_v07_file()` for all file lookups:
    `manifest.json`, `memories.db`, `schedules.json`, `stats_cache.json`
- Add one-shot auto-detection in `lsm/db/migration.py`:
  - `auto_detect_migration(global_folder: Path, config: LSMConfig) -> dict` with heuristics:
    1. `.chroma/` exists → `from_db = "chroma"`
    2. `data/lsm.db` or `.ingest/lsm.db` exists → `from_db = "sqlite"`, read
       `lsm_schema_versions` for `from_version`
    3. PostgreSQL configured and reachable → `from_db = "postgresql"`, read version from DB
    4. `manifest.json` in root or `.ingest/` without `lsm.db` →
       `from_version = "v0.7"` and `from_db = None` (legacy file mode)
    5. `to_db` = `config.db.provider`, `to_version` = `lsm.__version__`
  - Print detected parameters and proceed
- Update `lsm/ui/shell/cli.py` `run_migrate_cli()`:
  - Accept new parameters (`from_db`, `to_db`, `from_version`, `to_version`, `resume`)
  - Remove old `migration_source`/`migration_target` params
  - Call `auto_detect_migration()` when no explicit args provided

- Commit and push changes for this sub-phase.

**Files**:
- `lsm/__main__.py` — updated argument parser
- `lsm/db/migration.py` — v0.7 subfolder resolution, auto-detection
- `lsm/ui/shell/cli.py` — updated `run_migrate_cli()`
- `tests/test_vectordb/test_migration.py`:
  - Test: `_resolve_v07_file()` finds files in root, `.ingest/`, `Agents/`
  - Test: `auto_detect_migration()` detects `.chroma/` → `from_db = "chroma"`
  - Test: `auto_detect_migration()` detects `lsm.db` → reads schema version
  - Test: `auto_detect_migration()` detects `manifest.json` without DB → `from_version = "v0.7"`
  - Test: one-shot migration end-to-end with no explicit args

**Success criteria**: `lsm migrate` with no args auto-detects source and migrates. Explicit
`--from-db`/`--to-db`/`--from-version`/`--to-version` work correctly. v0.7 subfolder
resolution finds files in `.ingest/` and `Agents/`.

---

## 17.6: Migration Progress Tracking, Resume & Schema Evolution

**Description**: Add migration progress tracking with checkpoint support, resume for
interrupted migrations, and schema evolution to ensure the target database has all
current-version tables and columns after migration.

**Tasks**:
- Add migration progress tracking in `lsm/db/migration.py`:
  - New `table_names.migration_progress` table:
    ```sql
    CREATE TABLE IF NOT EXISTS <table_names.migration_progress> (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        migration_run TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        source_type TEXT NOT NULL,
        target_type TEXT NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        rows_processed INTEGER DEFAULT 0,
        error_message TEXT
    );
    ```
  - Stage names for copy phases: `'copy_vectors'`, `'copy_manifest'`, `'copy_memories'`,
    `'copy_memory_candidates'`, `'copy_schedules'`, `'copy_stats_cache'`,
    `'copy_remote_cache'`, `'copy_schema_versions'`
  - Enrichment stage names defined here but wired in 17.8: `'enrich_tier1_simhash'`,
    `'enrich_tier1_defaults'`, `'enrich_tier1_node_type'`, `'enrich_tier1_tags'`,
    `'enrich_tier2_heading_path'`, `'enrich_tier2_positions'`, `'enrich_tier2_graph'`,
    `'enrich_tier2_clusters'`, `'enrich_tier3_gap_detection'`
  - Wrap each copy stage with checkpoint tracking: before stage → insert `'in_progress'`;
    after → update to `'completed'`; on error → update to `'failed'` with error message
  - Add `_begin_stage(conn, run_id, stage, source_type, target_type) -> int` and
    `_complete_stage(conn, stage_id, rows=0)` and `_fail_stage(conn, stage_id, error)`
    helper functions for reuse across copy and enrichment stages
- Add resume support in `migrate()`:
  - When `--resume` flag is set: read `table_names.migration_progress` for the most recent
    `migration_run`, skip `'completed'` stages, re-run `'in_progress'`/`'failed'` stages,
    continue with `'pending'` stages
  - For vector migration: use `rows_processed` as the starting offset on resume
- Add schema evolution during migration:
  - After copying vectors and auxiliary state, call `ensure_application_schema()` on target
    to create any tables added in the current version
  - Add `_evolve_schema(target_conn, table_names)` for column-level evolution:
    add columns from newer versions. **SQLite limitation**: SQLite does not support
    `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Use `PRAGMA table_info()` to check
    existing columns before issuing `ALTER TABLE ... ADD COLUMN`. PostgreSQL supports
    `ADD COLUMN IF NOT EXISTS` directly.
  - Schema evolution runs **before** enrichment (17.7–17.8) so that target columns exist
    for backfill

- Commit and push changes for this sub-phase.

**Files**:
- `lsm/db/migration.py` — progress tracking, resume, schema evolution
- `tests/test_vectordb/test_migration.py`:
  - Test: migration progress records stages correctly (pending → in_progress → completed)
  - Test: failed stage records error message
  - Test: resume skips completed stages and re-runs failed/in-progress stages
  - Test: vector resume uses `rows_processed` offset
  - Test: schema evolution creates new tables on target after migration
  - Test: schema evolution adds new columns to existing tables

**Success criteria**: Every migration stage is tracked in `table_names.migration_progress`.
Interrupted migrations resume with `--resume`, skipping completed stages. Schema evolution
ensures target database has all current-version tables and columns.

---

## 17.7: Post-Migration Chunk Enrichment Pipeline

**Description**: Create the chunk enrichment module that brings existing chunks up to date
with the current ingest pipeline. This is a standalone module (`lsm/db/enrichment.py`)
that can be called from the migration flow (wired in 17.8) or independently.

### Background: The Chunk Enrichment Problem

The current migration copies chunks verbatim — ids, text, metadata, and embeddings are
transferred as-is with no transformation. This means chunks ingested under older pipeline
versions are missing fields and processing that newer phases introduced:

| Missing Field/Processing | Introduced By | Enrichment Tier |
|---|---|---|
| `simhash` (dedup fingerprint) | Phase 12 | Tier 1 — computed from `chunk_text` |
| `is_current` / `version` defaults | Phase 3 | Tier 1 — SQL defaults |
| `node_type` defaults (`"chunk"`) | Phase 14.1 | Tier 1 — SQL defaults + vec table sync |
| `root_tags` / `folder_tags` / `content_type` | Phase 3.3 | Tier 1 — from current config |
| `heading_path` (JSON heading hierarchy) | Phase 7.3 | Tier 2 — needs source file re-parse |
| `start_char` / `end_char` / `chunk_length` | Phase 1+ | Tier 2 — needs source file re-parse |
| Graph nodes/edges | Phase 15 | Tier 2 — needs source file re-parse |
| Changed chunk boundaries (heading depth) | Phase 7.1/7.2 | Tier 3 — full re-chunk + re-embed |
| Missing section/file summary chunks | Phase 14.1 | Tier 3 — LLM generation + embedding |
| Missing cluster assignments (`cluster_id`, `cluster_size`) when cluster retrieval is enabled | Phase 14.2 | Tier 2b — rebuild from existing embeddings (no re-ingest) |

The enrichment pipeline is organized into three tiers based on cost and disruption:

- **Tier 1 (in-place SQL updates)**: No re-parsing, no re-embedding. Pure functions of
  existing data or current config. Always applied automatically during migration.
- **Tier 2 (source-file re-parse)**: Needs access to the original source files. Backfills
  metadata without re-embedding. Applied automatically if source files are available;
  files without sources are logged as skipped.
- **Tier 2b (embedding-only recompute)**: Uses existing embeddings (no source files,
  no re-embedding) to rebuild derived per-chunk signals such as
  `cluster_id`/`cluster_size` when cluster retrieval is enabled.
- **Tier 3 (full re-ingest)**: Chunk boundaries or embeddings change. Cannot be done
  in-place. Migration reports which files need re-ingest and advises the user to run
  `lsm ingest --force-reingest-changed-config` after migration completes.

### Coverage Checklist Against PLAN.md

- Phase 3 (`PLAN.md`): backfill `version`, `is_current`, root/folder/content tags.
- Phase 7 (`PLAN.md`): backfill `heading_path`, `start_char`/`end_char`/`chunk_length`,
  detect chunk-boundary drift requiring re-ingest.
- Phase 12 (`PLAN.md`): backfill `simhash`.
- Phase 14 (`PLAN.md`): backfill `node_type="chunk"` defaults, detect missing section/file
  summary nodes, and rebuild per-chunk cluster assignments when enabled.
- Phase 15 (`PLAN.md`): backfill graph nodes/edges from source files.

### Tasks

- Create `lsm/db/enrichment.py` with:
  - `EnrichmentReport` dataclass:
    - `tier1_updated: int` — count of chunks updated by Tier 1
    - `tier2_updated: int` — count of chunks updated by Tier 2
    - `tier2b_updated: int` — count of chunks updated by Tier 2b (cluster rebuild)
    - `tier2_skipped: list[str]` — source paths not found on disk (Tier 2 skipped)
    - `tier3_needed: list[str]` — source paths that need full re-ingest (Tier 3)
    - `errors: list[str]` — any errors encountered
  - `detect_stale_chunks(conn, table_names) -> dict` — queries the target database to
    determine what enrichment is needed. Returns a dict keyed by tier:
    ```python
    {
        "tier1": {
            "simhash_null_count": int,     # chunks with NULL simhash
            "version_null_count": int,     # chunks with NULL version
            "node_type_null_count": int,   # chunks with NULL/empty node_type
            "tags_missing_count": int,     # chunks missing root_tags/content_type
        },
        "tier2": {
            "heading_path_null_count": int,  # chunks with heading but NULL heading_path
            "positions_null_count": int,     # chunks with NULL start_char
            "source_paths": list[str],       # distinct source_paths needing re-parse
            "cluster_rebuild_needed": bool,  # cluster_enabled and NULL cluster_id present
        },
        "tier3": {
            "schema_diff": dict,  # from check_schema_compatibility() — chunk_size,
                                  # chunking_strategy, or heading depth changes
            "missing_section_summary_files": list[str],
            "missing_file_summary_files": list[str],
            "needs_reingest": bool,
        },
    }
    ```
  - `run_tier1_enrichment(conn, config, table_names) -> int`:
    - **Simhash backfill**: Select all chunks where `simhash IS NULL`. For each batch,
      read `chunk_text`, call `compute_simhash(text)`, and `UPDATE` the `simhash` column.
      Uses batched reads (1000 chunks per batch) with progress logging.
    - **Version/is_current/node_type defaults**:
      - `UPDATE {table_names.chunks} SET version = 1 WHERE version IS NULL`
      - `UPDATE {table_names.chunks} SET is_current = 1 WHERE is_current IS NULL`
      - `UPDATE {table_names.chunks} SET node_type = 'chunk' WHERE node_type IS NULL OR node_type = ''`
      - Mirror `is_current` and `node_type` updates to `{table_names.vec_chunks}` so
        vector pre-filters stay consistent with `{table_names.chunks}`
    - **Tag enrichment**: For chunks where `root_tags IS NULL` or `content_type IS NULL`,
      use the current config's `RootConfig` entries to match `source_path` against
      configured roots and apply `root_tags`, `folder_tags`, and `content_type`. Reuses
      the existing `collect_folder_tags()` utility from the ingest pipeline.
    - Returns count of chunks updated
  - `run_tier2_enrichment(conn, config, table_names, root_paths) -> tuple[int, list[str]]`:
    - **Heading path backfill**: For each distinct `source_path` where at least one
      chunk-level row (`node_type = 'chunk'`) has
      `heading IS NOT NULL AND heading_path IS NULL`:
      1. Check if the source file still exists on disk
      2. If yes: re-parse the file using the appropriate parser (determined by extension),
         build a `FileGraph`, generate `heading_path` for each chunk by matching
         `chunk_index` and `heading` back to the graph hierarchy
      3. `UPDATE` the `heading_path` column for all matched chunks
      4. If no: add to `skipped` list
    - **Position backfill**: For chunk-level rows (`node_type = 'chunk'`) with
      `start_char IS NULL` where source exists:
      1. Re-parse and re-chunk the file using the current chunking pipeline
      2. Match chunks by `chunk_index` (the ordinal position within the file is stable
         if chunk boundaries haven't changed)
      3. `UPDATE` `start_char`, `end_char`, `chunk_length` from the re-chunked positions
      4. Verify match by comparing `chunk_text` prefix (first 100 chars) — skip if
         text doesn't match (indicates chunk boundaries changed → Tier 3)
    - **Graph backfill**: For source files that exist but have no entries in
      `{table_names.graph_nodes}`: re-parse the file, build `FileGraph`, extract internal
      links, and insert nodes/edges into the graph tables. Reuses
      `GraphBuilder.build_file_graph()`.
    - Returns `(updated_count, skipped_source_paths)`
  - `run_tier2_cluster_enrichment(conn, config, table_names) -> int`:
    - If `config.query.cluster_enabled` is `True`, call `build_clusters()` from
      `lsm/db/clustering.py` using configured algorithm/k.
    - Recompute and persist `cluster_id`/`cluster_size` for all chunk rows and centroids.
    - Sync `cluster_id` into `{table_names.vec_chunks}` for vec pre-filter parity.
    - Returns count of chunk rows updated.
  - `run_enrichment_pipeline(conn, config, table_names, root_paths, skip_tier2=False)
    -> EnrichmentReport`:
    - Orchestrator that runs `detect_stale_chunks()`, then Tier 1, then Tier 2 (unless
      `skip_tier2=True`), then Tier 2b cluster rebuild when needed, collects Tier 3
      recommendations (chunk-boundary drift and
      summary gaps), and returns the full report
    - Logs progress to stdout
    - Accepts optional `stage_tracker` callback (used by 17.8 to integrate with
      `table_names.migration_progress`)

- Commit and push changes for this sub-phase.

**Files**:
- `lsm/db/enrichment.py` — new post-migration chunk enrichment module:
  - `detect_stale_chunks()` — identifies what enrichment is needed per tier
  - `run_tier1_enrichment()` — simhash, version/is_current/node_type defaults, tag backfill
  - `run_tier2_enrichment()` — heading_path, positions, graph backfill
  - `run_tier2_cluster_enrichment()` — cluster_id/cluster_size rebuild from embeddings
  - `run_enrichment_pipeline()` — orchestrator
  - `EnrichmentReport` — structured results
- `tests/test_db/test_enrichment.py`:
  - Test: `detect_stale_chunks()` correctly counts NULL simhash, heading_path, etc.
  - Test: `detect_stale_chunks()` flags missing section/file summaries when enabled
  - Test: `detect_stale_chunks()` returns empty counts on a fully enriched database
  - Test: Tier 1 — simhash backfill computes correct simhash from chunk_text
  - Test: Tier 1 — version/is_current defaults applied to NULL rows
  - Test: Tier 1 — node_type defaults to `"chunk"` when NULL/empty
  - Test: Tier 1 — vec table `is_current`/`node_type` mirror updates stay in sync
  - Test: Tier 1 — tag enrichment matches source_path to configured roots
  - Test: Tier 1 — idempotent (running twice produces same result, no double-updates)
  - Test: Tier 2 — heading_path backfill re-parses source file and populates heading_path
  - Test: Tier 2 — position backfill updates start_char/end_char/chunk_length
  - Test: Tier 2 — position backfill skips chunks where text prefix doesn't match
    (chunk boundaries changed → tier 3 flag)
  - Test: Tier 2 — graph backfill creates nodes/edges for files without graph entries
  - Test: Tier 2 — missing source files added to skipped list, not errored
  - Test: Tier 2 — idempotent (chunks already enriched are not re-processed)
  - Test: Tier 2b — cluster rebuild repopulates `cluster_id`/`cluster_size`
  - Test: Tier 2b — vec table `cluster_id` stays in sync with chunk table
  - Test: full enrichment pipeline produces correct EnrichmentReport

**Success criteria**: `run_enrichment_pipeline()` correctly detects stale chunks across
Phase 3/7/12/14/15 requirements from `PLAN.md`, applies Tier 1 enrichments in-place,
applies Tier 2 enrichments when source files are available, applies Tier 2b cluster
rebuilds from existing embeddings when needed, reports Tier 3 re-ingest needs
(including summary gaps), and is fully idempotent. All enrichment functions are
independently testable without requiring a full migration.

---

## 17.8: Enrichment Integration & CLI Wiring

**Description**: Wire the enrichment pipeline (17.7) into the migration flow (17.5–17.6)
and add the `--enrich` / `--skip-enrich` CLI modes. This sub-phase connects the standalone
enrichment module to the migration system's progress tracking and CLI surface.

**Tasks**:
- Wire enrichment into migration flow in `lsm/db/migration.py`:
  - After `_copy_vectors()` + `_copy_auxiliary_state()` + `_evolve_schema()`, call
    `run_enrichment_pipeline()` unless `--skip-enrich` was passed
  - Ensure `run_enrichment_pipeline()` executes Tier 2b (`run_tier2_cluster_enrichment()`)
    when `detect_stale_chunks()["tier2"]["cluster_rebuild_needed"]` is true
  - Pass a `stage_tracker` callback that records each enrichment tier as stages in
    `table_names.migration_progress` (using the helpers from 17.6: `_begin_stage`,
    `_complete_stage`, `_fail_stage`)
  - On `--enrich` flag (standalone enrichment): skip the copy phase entirely, just run
    `run_enrichment_pipeline()` on the current database
  - Print a summary at the end:
    ```
    Migration complete.
      Chunks copied: 4,521
      Tier 1 enriched: 4,521 (simhash, version/is_current/node_type defaults, tags)
      Tier 2 enriched: 3,890 (heading_path, positions, graph)
      Tier 2b enriched: 4,521 (cluster_id/cluster_size rebuild)
      Tier 2 skipped:    631 (source files not found)
      Tier 3 needed:      14 (chunk boundary changes + missing summaries)
    ```
  - If Tier 3 is needed, print:
    ```
    WARNING: Some files need re-ingest (chunk boundaries changed and/or summaries missing).
    Run `lsm ingest --force-reingest-changed-config` to re-chunk, re-embed, and regenerate summaries.
    ```
- Wire `--enrich` and `--skip-enrich` flags in `lsm/__main__.py`:
  - `--enrich`: validate that no `--from-db`/`--to-db` are set (enrichment is in-place),
    dispatch to standalone enrichment path
  - `--skip-enrich`: pass flag through to `migrate()` to suppress enrichment after copy
  - Mutual exclusion: `--enrich` and `--skip-enrich` cannot be used together
- Update `lsm/ui/shell/cli.py` `run_migrate_cli()`:
  - Accept `enrich` and `skip_enrich` parameters
  - `--enrich` mode: call `run_enrichment_pipeline()` directly without backend migration
- Resume support for enrichment stages:
  - When `--resume` is used: enrichment stages in `table_names.migration_progress` follow the
    same resume logic as copy stages (skip completed, re-run failed/in-progress)
  - Enrichment is inherently idempotent so re-running completed stages is safe but
    unnecessary

- Commit and push changes for this sub-phase.

**Files**:
- `lsm/__main__.py` — `--enrich`/`--skip-enrich` flag wiring
- `lsm/db/migration.py` — enrichment integration after copy, standalone enrichment path
- `lsm/ui/shell/cli.py` — enrichment parameters in `run_migrate_cli()`
- `tests/test_db/test_enrichment.py` (additional integration tests):
  - Test: `--enrich` standalone mode enriches existing database without backend copy
  - Test: `--skip-enrich` skips enrichment after copy
  - Test: enrichment stages appear in `table_names.migration_progress` with correct status
  - Test: `--resume` after interrupted enrichment resumes from correct tier
  - Test: migration summary output includes enrichment counts
  - Test: Tier 3 warning printed when chunk boundaries or summary gaps are detected
  - Test: Tier 2b cluster rebuild runs when cluster retrieval is enabled and cluster IDs are missing
- `tests/test_vectordb/test_migration.py` (additional integration test):
  - Test: full migration end-to-end with enrichment (copy + enrich + summary)

**Success criteria**: `lsm migrate` with no args auto-detects source, copies data, and
enriches chunks. `lsm migrate --enrich` runs enrichment only on an existing database.
`--skip-enrich` skips enrichment for backend-only migrations. Enrichment stages are
tracked in `table_names.migration_progress` alongside copy stages. Interrupted enrichment
resumes with `--resume`. Summary output clearly reports Tier 1/Tier 2/Tier 2b enrichment,
what was skipped, and what still needs a full re-ingest.

---

## 17.9: Documentation and Changelog

**Description**: Update all user-facing documentation and changelog to reflect the changes
made in 17.2–17.8.

**Tasks**:
- Update `docs/CHANGELOG.md` with entries for:
  - Table name infrastructure and configurable `table_prefix`
  - Config restructure from `"vectordb"` to `"db"` with `"vector"` submodel
  - Database health check at startup (including stale chunk detection)
  - Migration CLI parameter restructure (`--from-db`/`--to-db`/`--from-version`/`--to-version`)
  - One-shot migration auto-detection
  - Migration resume support
  - Schema evolution during migration
  - Post-migration chunk enrichment pipeline (Tier 1/2/3)
  - `--enrich` and `--skip-enrich` flags
- Update `docs/user-guide/CONFIGURATION.md`:
  - Document `"db"` config section with `table_prefix`, `path`, connection fields, and
    `vector` submodel
  - Remove all `"vectordb"` references
- Update `docs/user-guide/VECTOR_DATABASES.md`:
  - Update config examples to use `"db"` key structure
  - Add section on migration enrichment: explain what happens when upgrading from an
    older version, what each tier does, and when `--force-reingest-changed-config` is
    needed vs. automatic enrichment
- Update `.agents/docs/architecture/` files as needed for DB layer changes
- Verify `example_config.json` uses the new `"db"` structure

- Commit and push changes for this sub-phase.

**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/CONFIGURATION.md`
- `docs/user-guide/VECTOR_DATABASES.md`
- `.agents/docs/architecture/` — relevant architecture docs
- `example_config.json` — verification

**Success criteria**: All documentation accurately reflects the new config structure,
migration CLI, startup health checks, and enrichment pipeline. No stale `"vectordb"`
references remain in docs.

---

## 17.10: Phase 17.2–17.9 Code Review

**Description**: Review all changes from 17.2–17.9 for correctness, completeness, and
code quality.

**Tasks**:
- Review all sub-phases and ensure there are no gaps or bugs remaining
- Review code for dead code or unnecessary complexity
- Review tests to ensure they are well-structured with no mocks or stubs on database
  operations
- Ensure PostgreSQL parity includes FTS coverage equivalent to SQLite-vec tests
  (explicitly add/verify tests around `query_text` / full-text search behavior)
- Explicitly close `17.1/16-C`: Phase 17 is not complete unless PostgreSQL FTS parity
  tests are added and passing
- Verify the full test suite passes: `.venv-wsl/bin/python -m pytest tests/ -v`
- Verify manual scenarios:
  - Load config with `"db"` key → works
  - Load config with old `"vectordb"` key → clear error message
  - Start app with `.chroma/` in global folder → health check warns `"legacy_detected"`
  - Start app with schema mismatch → health check warns `"mismatch"` with diff
  - Start app with stale chunks → health check warns `"stale_chunks"` (non-blocking)
  - `lsm migrate` with no args → auto-detects, migrates, and enriches
  - `lsm migrate --from-db chroma --to-db sqlite` → works, enrichment runs after copy
  - `lsm migrate --from-version v0.7 --to-db sqlite` → finds files in subfolders
  - `lsm migrate --enrich` → enriches existing database without backend copy
  - `lsm migrate --skip-enrich` → copies only, no enrichment
  - Interrupt migration during enrichment, re-run with `--resume` → resumes enrichment
  - After migration → target has all current-version tables
  - After migration → chunks have simhash, version, tags populated (Tier 1)
  - After migration → chunks with NULL/empty `node_type` are backfilled to `"chunk"`
    and vec-table metadata stays in sync
  - After migration → chunks with available source files have heading_path (Tier 2)
  - After migration → missing source files listed in summary as skipped
  - If chunk boundaries changed or summaries are missing → clear message advising
    `--force-reingest-changed-config`
  - If cluster retrieval is enabled and cluster assignments are missing → migration
    rebuilds clusters and reports Tier 2b counts
  - PostgreSQL provider full-text search behaves correctly and is covered by parity tests
- Confirm no new issues introduced

- Commit and push changes for this sub-phase.

**Success criteria**: Clean test suite. All debug items resolved. Manual verification
scenarios pass. Post-migration chunks are enriched to current pipeline standards.

---

*End of Phase 17.*
