# Testing Guide for Local Second Mind

This directory contains the test suite for Local Second Mind (LSM).

## Installation

Install test dependencies:

```bash
# Install package with development dependencies
pip install -e ".[dev]"

# Or install dependencies individually
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Run all tests:
```bash
pytest
```

### Run tests with coverage report:
```bash
pytest --cov=lsm --cov-report=term-missing
```

### Run specific test file:
```bash
pytest tests/test_ingest/test_parsers.py
```

### Run specific test class or function:
```bash
pytest tests/test_ingest/test_parsers.py::TestParseTxt
pytest tests/test_ingest/test_parsers.py::TestParseTxt::test_parse_txt_basic
```

### Run tests in verbose mode:
```bash
pytest -v
```

### Run tests and stop on first failure:
```bash
pytest -x
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_ingest/             # Ingest module tests
│   ├── test_parsers.py      # Parser function tests
│   ├── test_chunking.py     # Text chunking tests
│   ├── test_config.py       # Configuration tests
│   └── test_pipeline.py     # Pipeline integration tests (future)
├── test_query/              # Query module tests (future)
└── test_cli/                # CLI tests (future)
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- **Configuration:**
  - `sample_config_dict`: Minimal valid config dictionary
  - `sample_config_file`: Temporary JSON config file

- **File System:**
  - `sample_txt_file`: Sample .txt file
  - `sample_md_file`: Sample Markdown file
  - `sample_html_file`: Sample HTML file
  - `empty_file`: Empty file for edge cases
  - `document_root`: Document directory with nested structure

- **Mock Objects:**
  - `mock_openai_client`: Mocked OpenAI API client
  - `mock_embedder`: Mocked sentence-transformers model
  - `mock_chroma_collection`: Mocked ChromaDB collection

- **Environment:**
  - `clean_env`: Environment without API keys
  - `mock_env_with_api_key`: Environment with fake API key

## Coverage Goals

- **Phase 2 Target:** 60% code coverage
- **Long-term Goal:** 70%+ code coverage

Current coverage can be viewed after running:
```bash
pytest --cov=lsm --cov-report=html
open htmlcov/index.html  # View detailed coverage report
```

## Writing New Tests

### Test Naming Convention:
- Test files: `test_<module>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<what_is_being_tested>`

### Example Test:
```python
def test_feature_name(sample_fixture):
    """Brief description of what this test verifies."""
    # Arrange
    input_data = "test input"

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output
```

### Use Fixtures:
```python
def test_with_fixture(sample_config_file):
    """Use pre-configured fixtures from conftest.py."""
    config = load_config(sample_config_file)
    assert "roots" in config
```

### Mock External Dependencies:
```python
def test_with_mock(mock_openai_client):
    """Use mocked objects to avoid external API calls."""
    response = mock_openai_client.chat.completions.create(...)
    assert response is not None
```

## Best Practices

1. **Isolation:** Each test should be independent
2. **Clarity:** Test names should describe what's being tested
3. **Coverage:** Aim for edge cases and error conditions
4. **Speed:** Use mocks to avoid slow external calls
5. **Determinism:** Tests should produce consistent results

## Continuous Integration

Tests will be run automatically in CI/CD pipelines (future):
- On pull requests
- On commits to main branch
- Nightly builds

## Troubleshooting

### Import Errors:
```bash
# Make sure package is installed in editable mode
pip install -e .
```

### Coverage Not Showing:
```bash
# Install coverage plugin
pip install pytest-cov
```

### Tests Hanging:
- Check for infinite loops in code
- Ensure mocks are properly configured
- Look for blocking I/O operations

## Future Test Areas

- [ ] Manifest operations
- [ ] File system traversal
- [ ] Pipeline integration
- [ ] Query retrieval
- [ ] Reranking strategies
- [ ] CLI commands
- [ ] End-to-end workflows
