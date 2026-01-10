"""
Shared pytest fixtures for Local Second Mind tests.

This module provides reusable fixtures for testing the LSM application,
including mock configurations, sample documents, and temporary directories.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_config_dict(tmp_path: Path) -> Dict[str, Any]:
    """
    Provide a minimal valid configuration dictionary for testing.

    Uses tmp_path for all file system paths to ensure isolation.
    """
    return {
        "roots": [str(tmp_path / "documents")],
        "persist_dir": str(tmp_path / ".chroma"),
        "chroma_flush_interval": 2000,
        "collection": "test_kb",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "manifest": str(tmp_path / ".ingest" / "manifest.json"),
        "extensions": [".txt", ".md", ".pdf"],
        "override_extensions": False,
        "exclude_dirs": [".cache"],
        "override_excludes": False,
        "dry_run": False,
        "openai": {
            "api_key": None  # Use environment variable
        },
        "query": {
            "k": 12,
            "k_rerank": 6,
            "no_rerank": False,
            "max_per_file": 2,
            "local_pool": 36,
            "model": "gpt-5.2",
            "min_relevance": 0.25,
            "retrieve_k": 36
        }
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config_dict: Dict[str, Any]) -> Path:
    """
    Create a temporary JSON config file for testing.

    Returns the path to the created config file.
    """
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(sample_config_dict, indent=2))
    return config_path


# -----------------------------------------------------------------------------
# File System Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """
    Create a sample .txt file for testing parsers.
    """
    content = """This is a sample text file for testing.

It contains multiple paragraphs to test chunking and parsing functionality.

The content is simple but sufficient for unit tests."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """
    Create a sample Markdown file for testing parsers.
    """
    content = """# Test Document

This is a **sample** Markdown file.

## Section 1

Some content in section 1.

## Section 2

Some content in section 2 with a [link](https://example.com).

### Subsection 2.1

- Item 1
- Item 2
- Item 3
"""

    file_path = tmp_path / "sample.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    """
    Create a sample HTML file for testing parsers.
    """
    content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <h1>Test Document</h1>
    <p>This is a sample HTML file for testing.</p>
    <p>It contains <strong>formatted</strong> text.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
</body>
</html>
"""

    file_path = tmp_path / "sample.html"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """
    Create an empty file for testing edge cases.
    """
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def document_root(tmp_path: Path, sample_txt_file: Path, sample_md_file: Path) -> Path:
    """
    Create a document root directory with sample files.

    Structure:
        documents/
            sample.txt
            sample.md
            subdir/
                nested.txt
    """
    doc_root = tmp_path / "documents"
    doc_root.mkdir(exist_ok=True)

    # Copy sample files to doc root
    (doc_root / "sample.txt").write_text(sample_txt_file.read_text())
    (doc_root / "sample.md").write_text(sample_md_file.read_text())

    # Create subdirectory with nested file
    subdir = doc_root / "subdir"
    subdir.mkdir(exist_ok=True)
    (subdir / "nested.txt").write_text("Nested file content for testing.")

    return doc_root


# -----------------------------------------------------------------------------
# Mock Object Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client(mocker):
    """
    Provide a mocked OpenAI client for testing query functionality.

    Prevents actual API calls during tests.
    """
    mock_client = mocker.MagicMock()

    # Mock chat completions response
    mock_response = mocker.MagicMock()
    mock_response.choices = [
        mocker.MagicMock(
            message=mocker.MagicMock(
                content="This is a mocked LLM response with citations [S1]."
            )
        )
    ]

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_embedder(mocker):
    """
    Provide a mocked sentence-transformers model for testing.

    Prevents loading actual models during tests.
    """
    mock_model = mocker.MagicMock()

    # Mock encode method to return dummy embeddings
    def mock_encode(texts, **kwargs):
        import numpy as np
        # Return fake embeddings (384-dim for all-MiniLM-L6-v2)
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 384).astype(np.float32)

    mock_model.encode = mock_encode

    return mock_model


@pytest.fixture
def mock_chroma_collection(mocker):
    """
    Provide a mocked ChromaDB collection for testing.

    Prevents actual database operations during tests.
    """
    mock_collection = mocker.MagicMock()

    # Mock query method
    mock_collection.query.return_value = {
        "ids": [["chunk_1", "chunk_2"]],
        "distances": [[0.1, 0.2]],
        "documents": [["Sample chunk 1", "Sample chunk 2"]],
        "metadatas": [[
            {"source_path": "/path/to/doc1.txt", "ext": ".txt"},
            {"source_path": "/path/to/doc2.txt", "ext": ".txt"}
        ]]
    }

    # Mock add method
    mock_collection.add.return_value = None

    # Mock count method
    mock_collection.count.return_value = 0

    return mock_collection


# -----------------------------------------------------------------------------
# Environment Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    """
    Provide a clean environment without API keys.

    Useful for testing environment variable handling.
    """
    # Remove OpenAI API key if it exists
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return monkeypatch


@pytest.fixture
def mock_env_with_api_key(monkeypatch):
    """
    Provide an environment with a fake OpenAI API key set.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
    return monkeypatch


# -----------------------------------------------------------------------------
# Data Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    """
    Provide sample text chunks for testing chunking and embedding.
    """
    return [
        "This is the first chunk of text. It should be around 200 characters to test chunking behavior properly.",
        "This is the second chunk. It overlaps with the first chunk by a certain number of characters.",
        "Third chunk here with different content to ensure diversity in test data.",
    ]


@pytest.fixture
def sample_metadata():
    """
    Provide sample metadata for testing chunk storage.
    """
    return [
        {
            "source_path": "/path/to/doc1.txt",
            "source_name": "doc1.txt",
            "ext": ".txt",
            "mtime_ns": 1704067200000000000,
            "file_hash": "abc123",
            "chunk_index": 0,
            "ingested_at": "2024-01-01T00:00:00Z"
        },
        {
            "source_path": "/path/to/doc2.md",
            "source_name": "doc2.md",
            "ext": ".md",
            "mtime_ns": 1704153600000000000,
            "file_hash": "def456",
            "chunk_index": 0,
            "ingested_at": "2024-01-02T00:00:00Z"
        }
    ]
