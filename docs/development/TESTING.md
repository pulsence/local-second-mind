# Testing Guide

This guide summarizes the test strategy and how to run tests for LSM.

## Philosophy

- Prefer unit tests for deterministic logic.
- Use fixtures for filesystem and config setup.
- Mock external services (OpenAI, Brave, Chroma) in tests.
- Keep tests fast and isolated.

## Install Test Dependencies

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=lsm --cov-report=term-missing
```

Single test file:

```bash
pytest tests/test_ingest/test_parsers.py
```

Single test function:

```bash
pytest tests/test_ingest/test_parsers.py::TestParseTxt::test_parse_txt_basic
```

## Test Layout

```
tests/
  conftest.py
  test_ingest/
  test_query/
  test_providers/
  test_cli/
  performance/
```

## Fixtures

Key fixtures live in `tests/conftest.py`:

- sample config dict and files
- test documents and directories
- mocked OpenAI client and embedder
- mocked Chroma collection

## Writing Tests

Guidelines:

- Use `test_<module>.py` filenames.
- Use `Test<FeatureName>` classes for grouping.
- Use explicit arrange / act / assert structure.

Example:

```python
def test_parse_txt_basic(sample_txt_file):
    text, meta = parse_txt(sample_txt_file)
    assert "hello" in text
```

## Coverage Goals

- Short term: 60% coverage
- Long term: 70%+ coverage

## Performance Tests

Performance tests live under `tests/performance`. Run them selectively:

```bash
pytest tests/performance/test_large_scale.py
```

## Troubleshooting

- Import errors: run `pip install -e .`
- Missing dependencies: check `pyproject.toml` extras
- Network access: tests should use mocks, not real API calls
