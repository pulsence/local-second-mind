# Testing Guide

This guide defines the tiered test strategy for LSM v0.5.0 development.

## Test Tiers

- `smoke`: Fast tests with lightweight fakes, no live services.
- `integration`: Real embeddings, real ChromaDB, real file parsing, no external network dependencies.
- `live`: Real external API calls.
- `live_llm`: Live subset for LLM providers.
- `live_remote`: Live subset for remote source providers.
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

```bash
# Default run (excludes live/docker)
pytest tests/ -v

# Only live tests
pytest tests/ -v -m "live"

# Everything including live/docker
pytest tests/ -v -m ""

# Only live LLM tests
pytest tests/ -v -m "live_llm"

# Only live remote provider tests
pytest tests/ -v -m "live_remote"
```

## Infrastructure Validation

Configuration loader behavior is covered by:

- `tests/test_infrastructure/test_test_config.py`

## Full Pipeline Suite

- `tests/test_integration/test_full_pipeline.py`

It includes:

- `@pytest.mark.integration` end-to-end ingest + retrieval (no external network calls)
- `@pytest.mark.live` end-to-end ingest + LLM rerank + synthesis
- `@pytest.mark.performance` scale/latency check with 100+ chunks (gated by `LSM_PERF_TEST`)

Examples:

```bash
# End-to-end ingest + retrieval without live services
pytest tests/test_integration/test_full_pipeline.py -v -m "integration"

# Live full pipeline (requires configured LSM_TEST_* provider credentials)
pytest tests/test_integration/test_full_pipeline.py -v -m "live"

# Performance scenario
LSM_PERF_TEST=1 pytest tests/test_integration/test_full_pipeline.py -v -m "performance"
```
