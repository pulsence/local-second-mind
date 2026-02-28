# Phase 17: Final Code Review and Release

**Status**: Pending

Comprehensive code review, integration testing, documentation finalization, and release
preparation.

---

## 17.1: Comprehensive Code Review

**Tasks**:
- Review all changes for backwards compatibility issues — ensure no unintended regressions
- Review for deprecated code, dead code, or legacy compatibility shims — remove them
- Review all SQL queries for injection risks — parameterized queries only
- Review all new modules for proper error handling and logging
- Review import graph — no circular dependencies
- Review test suite:
  - No mocks or stubs on database operations
  - No auto-pass tests
  - Test structure matches project module structure
  - All new features have unit + integration tests
- Review security:
  - FTS5 queries use parameterized queries
  - sqlite-vec queries use `?` placeholders
  - Cross-encoder inputs are sanitized
  - Graph traversal depth is bounded
  - Agent tools validate permissions correctly

- Commit and push changes for this sub-phase.
**Files**: All modified files from all phases

**Success criteria**: No dead code, no injection risks, no circular dependencies, no
auto-pass tests.

---

## 17.2: Integration Testing

**Tasks**:
- Run end-to-end integration tests:
  - Full ingest → query → answer cycle with SQLite provider
  - Full ingest → query → answer cycle with PostgreSQL provider
  - Migration between providers preserves all data
  - All five retrieval profiles produce results
  - Agent tools work through full pipeline
  - Agent multi-turn query chain uses returned `response_id` / `conversation_id`
  - Agent `_run_phase()` / `run_bounded()` multi-context runs preserve independent
    `context_label` + conversation-chain state
  - Multi-hop retrieval produces answers
  - Eval harness runs against real corpus
  - TUI chat flow preserves and advances conversation chaining across turns
  - Mode/model switch resets conversation chain deterministically
  - TUI Help screen shows updated `WHAT'S NEW` content for v0.8.0
- Run performance tests and assert against SLO targets from §6.8:
  - TUI startup SLA
  - Retrieval stage p95 ≤ 500ms (at 100k chunks, 384 dims)
  - End-to-end query p95 ≤ 1.5s for local interactive use
  - Query latency at 10k, 50k, 100k chunks
  - Ingest throughput
- Run all existing tests: `pytest tests/ -v --cov=lsm --cov-report=html`

- Commit and push changes for this sub-phase.
**Files**:
- `tests/test_integration/` — end-to-end tests
- `tests/performance/` — performance tests

**Success criteria**: All integration tests pass. Performance is within documented SLO
targets. Coverage report generated.

---

## 17.3: Architecture Documentation Update

**Tasks**:
- Update all `.agents/docs/architecture/` files to reflect v0.8.0 changes:
  - `development/OVERVIEW.md` — updated system map
  - `development/INGEST.md` — FileGraph integration, multi-vector, graph construction
  - `development/QUERY.md` — RetrievalPipeline, profiles, stages
  - `development/PROVIDERS.md` — simplified provider interface
  - `development/MODES.md` — updated ModeConfig
  - `development/AGENTS.md` — new pipeline tools, DB-backed memory/schedules
  - `packages/lsm.vectordb.md` — SQLite-vec provider, removed ChromaDB
  - `packages/lsm.query.md` — pipeline, stages, types
  - `packages/lsm.ingest.md` — heading enhancements, graph builder, multi-vector
  - `packages/lsm.providers.md` — simplified interface
  - `api-reference/CONFIG.md` — all config changes
  - `api-reference/PROVIDERS.md` — new send_message signature
- Update `ARCHITECTURE.md` top-level overview

- Commit and push changes for this sub-phase.
**Files**: All architecture docs listed above

**Success criteria**: Architecture docs accurately reflect v0.8.0 codebase.

---

## 17.4: Release Preparation

**Tasks**:
- Finalize `docs/CHANGELOG.md` with complete v0.8.0 entry
- Update version in `pyproject.toml` to `0.8.0`
- Update `README.md`
- Verify `example_config.json` is complete and correct
- Verify `.env.example` is complete
- Update TUI release notes in `lsm/ui/tui/screens/help.py`:
  - Refresh `WHAT'S NEW` entries for v0.8.0 feature highlights
  - Keep entries synchronized with release notes/changelog language
- Update `tests/test_ui/tui/test_screens.py` to validate v0.8.0 `WHAT'S NEW` content
- Update `docs/user-guide/AGENTS.md`:
  - Add a `Tools Access` subsection for each agent
  - Enumerate the exact tools each agent can invoke and any scope limits
- Create upgrade guide for v0.7.x → v0.8.0 users:
  - Config file changes required
  - `lsm migrate` instructions
  - Breaking changes summary
- Run final full test suite: `pytest tests/ -v`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `pyproject.toml`
- `README.md`
- `example_config.json`
- `.env.example`
- `lsm/ui/tui/screens/help.py`
- `tests/test_ui/tui/test_screens.py`
- `docs/user-guide/AGENTS.md`

**Success criteria**: All tests pass. Version bumped. Changelog complete. Upgrade guide
written. TUI `WHAT'S NEW` reflects v0.8.0. `AGENTS.md` documents per-agent tool access.
Ready for release.

---

*End of Phase 17.*
