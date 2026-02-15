# Testing Guide

This guide reflects the v0.5.0 tiered test strategy and runtime configuration.

## Test Tiers

- `smoke`: Fast tests with lightweight fakes, no live services.
- `integration`: Real embeddings, real ChromaDB, real file parsing, no external network dependencies.
- `live`: Real external API calls.
- `live_llm`: Live subset for LLM providers.
- `live_remote`: Live subset for remote source providers.
- `live_vectordb`: Live subset for PostgreSQL/pgvector and migration tests.
- `docker`: Tests that require Docker runtime support.
- `performance`: Scale/perf scenarios (run explicitly).

## Default Marker Behavior

`pytest` now defaults to excluding `live` and `docker` tests via:

```text
-m "not live and not docker"
```

This means `pytest tests/ -v` runs smoke + integration + performance-marked tests that are not also `live`/`docker`.

## Test Runtime Configuration

Test runtime settings are loaded from:

- `tests/.env.test` (local, gitignored)
- `LSM_TEST_*` environment variables
- `tests/testing_config.py` loader helpers

Template file:

- `tests/.env.test.example`

Supported variables:

- `LSM_TEST_CONFIG`
- `LSM_TEST_OPENAI_API_KEY`
- `LSM_TEST_ANTHROPIC_API_KEY`
- `LSM_TEST_GOOGLE_API_KEY`
- `LSM_TEST_OLLAMA_BASE_URL`
- `LSM_TEST_BRAVE_API_KEY`
- `LSM_TEST_SEMANTIC_SCHOLAR_API_KEY`
- `LSM_TEST_CORE_API_KEY`
- `LSM_TEST_POSTGRES_CONNECTION_STRING`
- `LSM_TEST_EMBED_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `LSM_TEST_TIER` (`smoke|integration|live`, default: `smoke`)

## Core Infrastructure Fixtures

The following fixtures are available in `tests/conftest.py`:

- `test_config`
- `real_embedder`
- `real_chromadb_provider`
- `real_openai_provider`
- `real_anthropic_provider`
- `real_gemini_provider`
- `real_local_provider`
- `real_postgresql_provider`
- `rich_test_corpus`
- `populated_chromadb`

Live provider fixtures auto-skip when required credentials are not configured.

## Synthetic Data Corpus

Comprehensive synthetic fixtures live under `tests/fixtures/synthetic_data/`:

- `documents/philosophy_essay.txt` (long-form prose corpus)
- `documents/research_paper.md` (markdown paper with headings, lists, code, references)
- `documents/technical_manual.html` (nested HTML with tables/lists)
- `documents/large_document.md` (stress-test large markdown corpus)
- `documents/short_note.txt`, `documents/empty_with_whitespace.txt`, `documents/unicode_content.txt`
- `documents/duplicate_content_1.txt`, `documents/duplicate_content_2.txt`, `documents/near_duplicate.txt`
- `documents/nested/` with `.lsm_tags.json` files and nested notes
- `configs/test_config_openai.json`, `configs/test_config_local.json`, `configs/test_config_minimal.json`

Validation test:

- `tests/test_fixtures/test_synthetic_data.py`

## Running Tests

```powershell
# Default run (excludes live/docker)
.venv\Scripts\python -m pytest tests/ -v

# Only live tests
.venv\Scripts\python -m pytest tests/ -v -m "live"

# Everything including live/docker
.venv\Scripts\python -m pytest tests/ -v -m ""

# Only live LLM tests
.venv\Scripts\python -m pytest tests/ -v -m "live_llm"

# Only live remote provider tests
.venv\Scripts\python -m pytest tests/ -v -m "live_remote"

# Only live vector DB tests (PostgreSQL + migration)
.venv\Scripts\python -m pytest tests/ -v -m "live_vectordb"
```

## TUI Tests

TUI tests use lightweight fake widget doubles instead of full Textual app mounts.

### Test Organization

```
tests/test_ui/
  test_command_parsing_contracts.py  # Command grammar contract tests
  helpers/
    test_fixtures_sanity.py          # Shared fixture validation
  tui/
    fixtures/                        # Shared test doubles and factories
      widgets.py                     # FakeStatic, FakeInput, FakeSelect, etc.
      app.py                         # create_fake_app() factory
    presenters/                      # Presenter-focused tests
      test_agents_presenters.py      # Agent log formatting tests
    test_base_screen_mixin.py        # Shared lifecycle mixin tests
    test_layout_structure.py         # Widget composition structure tests
    test_compact_layout.py           # Density/CSS structure tests
    test_settings_screen.py          # Settings behavior tests
    test_agents_screen.py            # Agents behavior + binding tests
    test_remote_screen.py            # Remote behavior + binding tests
    test_presenters.py               # Presenter unit tests
    test_startup_smoke.py            # Startup crash-free launch tests
    test_performance.py              # Startup performance budget tests
```

### Test Categories

- **Contract tests** - Verify parsing grammar and command validation
- **Structure tests** - Verify widget composition and CSS class presence
- **Behavior tests** - Verify interaction flows through fake doubles
- **Binding tests** - Verify keybinding sets and conflict-freedom
- **Presenter tests** - Verify pure formatting functions independently
- **Mixin tests** - Verify shared lifecycle helpers
- **Smoke tests** - Verify crash-free startup and screen imports
- **Performance tests** - Verify startup timing budget and milestone recording

### Naming Conventions

- `test_<screen>_screen.py` - Screen-level behavior tests
- `test_<feature>_contracts.py` - Grammar/protocol contract tests
- `test_<module>_presenters.py` - Presenter formatting tests
- `test_<concept>.py` - Cross-cutting concept tests

See the [TUI Architecture Guide](TUI_ARCHITECTURE.md) for full conventions.

### Startup Smoke Tests

`tests/test_ui/tui/test_startup_smoke.py` verifies crash-free TUI startup:

- `LSMApp.__init__` completes without raising
- `compose()` references all 5 screen classes and tab IDs
- `on_mount()` reaches Query context as default home screen
- Startup survives failure of non-critical components (logging, agent binding)
- All screen classes are importable without error
- Providers remain lazy (not initialized at startup)

These tests run in the default test suite (no `live` or `docker` markers).

### Startup Performance Budget

`tests/test_ui/tui/test_performance.py` enforces startup timing:

- **SLA**: Query/Home interactive in < 1.0 second
- **Threshold override**: Set `LSM_TEST_STARTUP_BUDGET_MS` env var for CI
- Tests verify `StartupTimeline` milestone recording, ordering, and budget compliance
- Tests verify non-critical work (agent runtime binding) is deferred to background
- Tests verify `AgentsScreen` deferred initialization runs on first activation

```bash
# Run startup performance tests
.venv-wsl/bin/python -m pytest tests/test_ui/tui/test_performance.py -v

# Override budget threshold for slow CI
LSM_TEST_STARTUP_BUDGET_MS=2000 .venv-wsl/bin/python -m pytest tests/test_ui/tui/test_performance.py -v
```

## Security Suite

Agent sandbox security tests are adversarial and should run as part of default validation when changing `lsm/agents/` or `lsm/agents/tools/`.

```powershell
# Run all STRIDE security suites (T1-T8)
.venv\Scripts\python -m pytest tests/test_agents/test_security_*.py -v

# Run full agent test suite
.venv\Scripts\python -m pytest tests/test_agents -v
```

Security suite files:

- `tests/test_agents/test_security_paths.py`
- `tests/test_agents/test_security_permissions.py`
- `tests/test_agents/test_security_network.py`
- `tests/test_agents/test_security_resources.py`
- `tests/test_agents/test_security_integrity.py`
- `tests/test_agents/test_security_injection.py`
- `tests/test_agents/test_security_secrets.py`

## Infrastructure Validation

Configuration loader behavior is covered by:

- `tests/test_infrastructure/test_test_config.py`

## PostgreSQL + Migration Coverage

Live PostgreSQL provider and migration tests:

- `tests/test_vectordb/test_live_postgresql.py`
- `tests/test_vectordb/test_live_migration_chromadb_to_postgres.py`

These validate:

- Real CRUD/query/filter/update/delete behavior on PostgreSQL + pgvector
- Pagination and stats on a live PostgreSQL collection
- End-to-end migration from a real ChromaDB collection into PostgreSQL
- Post-migration retrieval and semantic query behavior

Prerequisites for live PostgreSQL tests:

- `LSM_TEST_POSTGRES_CONNECTION_STRING` points to a reachable database
- Current DB role has `CREATE` privilege on schema `public`
- `pgvector` extension exists (or the role can create it)

## Full Pipeline Suite

- `tests/test_integration/test_full_pipeline.py`

It includes:

- `@pytest.mark.integration` end-to-end ingest + retrieval (no external network calls)
- `@pytest.mark.live` end-to-end ingest + LLM rerank + synthesis
- `@pytest.mark.live` + `@pytest.mark.live_vectordb` end-to-end ingest + retrieval using PostgreSQL/pgvector as the vector store
- `@pytest.mark.performance` scale/latency check with 100+ chunks (gated by `LSM_PERF_TEST`)

Examples:

```powershell
# End-to-end ingest + retrieval without live services
.venv\Scripts\python -m pytest tests/test_integration/test_full_pipeline.py -v -m "integration"

# Live full pipeline (requires configured LSM_TEST_* provider credentials)
.venv\Scripts\python -m pytest tests/test_integration/test_full_pipeline.py -v -m "live"

# Performance scenario
$env:LSM_PERF_TEST=1; .venv\Scripts\python -m pytest tests/test_integration/test_full_pipeline.py -v -m "performance"
```
