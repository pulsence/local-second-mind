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
- `mock_openai_client`, `mock_embedder`, `mock_chroma_collection`
- `mock_vectordb_provider`
- `ingest_config`
- `progress_callback_mock`
- `global_folder`
- `clean_env`, `mock_env_with_api_key`
- `sample_chunks`, `sample_metadata`

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

- Some suites depend on optional third-party packages or mocked network responses.
- If full-suite failures occur, validate whether they are in-scope for the current phase before fixing.
