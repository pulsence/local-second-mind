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
