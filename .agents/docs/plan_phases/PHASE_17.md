# Phase 17: Debug Phase

**Status**: Pending

Pre-release debug phase for resolving implementation gaps and bugs identified during the
Phase 11–16 review, plus any additional issues surfaced by the user.

---

## 17.1: Issues Identified During Phase 11–16 Review

The following issues were cataloged during a comprehensive review of phases 11–16. Each
item includes severity, affected phase, location, and description.

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
| 16-C | Major | Missing PostgreSQL parity tests: no FTS tests, no graph tests, no prune tests. Phase 16.1 success criteria requires PostgreSQL provider to pass the same interface tests as SQLite-vec. | `tests/test_vectordb/test_postgresql.py` |
| 16-D | Medium | CLI post-ingest advisories not implemented. Phase 16.2 specifies "Also emit advisories after `lsm ingest` on CLI path." | `lsm/ui/shell/cli.py`, `lsm/__main__.py` |

---

## 17.2: User-Reported Debug Errors

_This section is reserved for additional issues identified by the user during manual
testing and review. Items will be added here before implementation begins._

---

## 17.3: Fix Implementation

**Tasks**:
- Triage all items from 17.1 and 17.2
- Implement fixes for all accepted items
- Add or update tests for each fix
- Run full test suite after each fix batch

- Commit and push changes for this sub-phase.

**Success criteria**: All accepted issues resolved. Full test suite passes. No regressions.

---

## 17.4: Debug Phase Review

**Tasks**:
- Verify all fixes are correct and complete
- Run full test suite: `.venv-wsl/bin/python -m pytest tests/ -v`
- Confirm no new issues introduced

- Commit and push changes for this sub-phase.

**Success criteria**: Clean test suite. All debug items resolved or explicitly deferred
to a future release.

---

*End of Phase 17.*
