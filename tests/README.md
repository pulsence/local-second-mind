# Testing Guide for Local Second Mind

This directory contains unit, integration, and performance tests for Local Second Mind.

## Quick Start

```bash
# Install package with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=lsm --cov-report=term-missing
```


## Test Layout

```text
tests/
  conftest.py
  test_config/
  test_ingest/
  test_query/
  test_providers/
  test_ui/
  test_vectordb/
  performance/
```

## Key Fixtures (`tests/conftest.py`)

- `sample_config_dict`, `sample_config_file`
- `sample_txt_file`, `sample_md_file`, `sample_html_file`, `empty_file`, `document_root`
- `ingest_config`
- `global_folder`
- `clean_env`, `mock_env_with_api_key`
- `sample_chunks`, `sample_metadata`
- `synthetic_data_root`
- `test_config`
- `real_embedder`, `real_chromadb_provider`, `real_postgresql_provider`
- `real_openai_provider`, `real_anthropic_provider`, `real_gemini_provider`, `real_local_provider`
- `rich_test_corpus`, `populated_chromadb`

## Useful Commands

```bash
# Run only VectorDB tests
pytest tests/test_vectordb -v

# Run Phase 6 ingest tests
pytest tests/test_ingest/test_api.py tests/test_ingest/test_manifest.py tests/test_ingest/test_fs.py -v

# Run integration tests (Phase 7)
pytest tests/ -v -m integration

# Skip integration/performance suites
pytest tests/ -v -m "not integration and not performance"

# Run live PostgreSQL vector DB tests
pytest tests/test_vectordb/test_live_postgresql.py tests/test_vectordb/test_live_migration_chromadb_to_postgres.py -v -m "live"
```

## Synthetic Fixtures

Synthetic integration fixtures live in `tests/fixtures/synthetic_data/`:
- `sample.txt`
- `sample.md`
- `sample.html`
- `nested/nested_note.txt`
- `documents/` (long-form corpus, edge cases, duplicates, nested tags)
- `configs/` (openai/local/minimal test config templates)

Integration suites using these fixtures:
- `tests/test_ingest/test_integration.py`
- `tests/test_query/test_integration_progress.py`
- `tests/test_fixtures/test_synthetic_data.py`

## Notes

- Some suites depend on optional third-party packages.
- Unit tests still use targeted mocks for SDK/network failure paths that cannot be reproduced deterministically.
- If full-suite failures occur, validate whether they are in-scope for the current phase before fixing.
