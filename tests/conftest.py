"""
Shared pytest fixtures for Local Second Mind tests.

This module provides reusable fixtures for testing the LSM application,
including mock configurations, sample documents, and temporary directories.
"""

import json
import shutil
import pytest
from pathlib import Path
from typing import Dict, Any
from uuid import uuid4

from lsm.config.models import (
    GlobalConfig,
    IngestConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    QueryConfig,
    VectorDBConfig,
)
from tests.testing_config import TestConfig, load_test_config


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
        "global": {
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        },
        "ingest": {
            "roots": [str(tmp_path / "documents")],
            "persist_dir": str(tmp_path / ".chroma"),
            "chroma_flush_interval": 2000,
            "collection": "test_kb",
            "manifest": str(tmp_path / ".ingest" / "manifest.json"),
            "extensions": [".txt", ".md", ".pdf"],
            "override_extensions": False,
            "exclude_dirs": [".cache"],
            "override_excludes": False,
            "dry_run": False,
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "test_kb",
        },
        "llms": {
            "providers": [
                {"provider_name": "openai", "api_key": None}
            ],
            "services": {
                "query": {"provider": "openai", "model": "gpt-5.2"}
            },
        },
        "query": {
            "k": 12,
            "k_rerank": 6,
            "no_rerank": False,
            "max_per_file": 2,
            "local_pool": 36,
            "min_relevance": 0.25,
            "retrieve_k": 36,
        },
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
    file_path = tmp_path / "sample.txt"
    source = Path(__file__).parent / "fixtures" / "synthetic_data" / "sample.txt"
    file_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return file_path


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """
    Create a sample Markdown file for testing parsers.
    """
    file_path = tmp_path / "sample.md"
    source = Path(__file__).parent / "fixtures" / "synthetic_data" / "sample.md"
    file_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return file_path


@pytest.fixture
def sample_html_file(tmp_path: Path) -> Path:
    """
    Create a sample HTML file for testing parsers.
    """
    file_path = tmp_path / "sample.html"
    source = Path(__file__).parent / "fixtures" / "synthetic_data" / "sample.html"
    file_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
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


@pytest.fixture
def global_folder(tmp_path: Path, monkeypatch) -> Path:
    """
    Provide an isolated global folder for tests.
    """
    folder = tmp_path / "global"
    monkeypatch.setenv("LSM_GLOBAL_FOLDER", str(folder))
    return folder


@pytest.fixture
def ingest_config(tmp_path: Path, global_folder: Path) -> LSMConfig:
    """
    Provide a minimal valid LSMConfig for ingest API tests.
    """
    roots_dir = tmp_path / "docs"
    roots_dir.mkdir(parents=True, exist_ok=True)

    ingest = IngestConfig(
        roots=[roots_dir],
        persist_dir=tmp_path / ".chroma",
        collection="test_collection",
        manifest=tmp_path / ".ingest" / "manifest.json",
    )
    query = QueryConfig()
    llm = LLMRegistryConfig(
        providers=[LLMProviderConfig(provider_name="local")],
        services={"query": LLMServiceConfig(provider="local", model="llama3.1")},
    )
    vectordb = VectorDBConfig(
        provider="chromadb",
        persist_dir=tmp_path / ".chroma",
        collection="test_collection",
    )
    return LSMConfig(
        ingest=ingest,
        query=query,
        llm=llm,
        vectordb=vectordb,
        global_settings=GlobalConfig(global_folder=global_folder),
    )


@pytest.fixture
def synthetic_data_root(tmp_path: Path) -> Path:
    """
    Copy static synthetic integration fixtures into a temp directory.
    """
    source = Path(__file__).parent / "fixtures" / "synthetic_data"
    dest = tmp_path / "synthetic_data"
    shutil.copytree(source, dest)
    return dest


# -----------------------------------------------------------------------------
# Tier-Aware Test Infrastructure Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """
    Provide resolved test runtime configuration from `LSM_TEST_*`.
    """
    return load_test_config()


@pytest.fixture(scope="session")
def live_postgres_connection_string(test_config: TestConfig) -> str:
    """
    Return PostgreSQL DSN for live vector DB tests after preflight checks.

    Preconditions:
    - `LSM_TEST_POSTGRES_CONNECTION_STRING` is configured
    - connection succeeds
    - current user can create tables in `public` schema
    - pgvector extension exists (or can be created)
    """
    dsn = test_config.postgres_connection_string
    if not dsn:
        pytest.skip(
            "Set LSM_TEST_POSTGRES_CONNECTION_STRING to enable live PostgreSQL vector DB tests"
        )

    try:
        import psycopg2
    except Exception as exc:
        pytest.skip(f"psycopg2 is unavailable for PostgreSQL live tests: {exc}")

    try:
        conn = psycopg2.connect(dsn)
    except Exception as exc:
        pytest.skip(f"Unable to connect to PostgreSQL test database: {exc}")

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        current_user,
                        has_schema_privilege(current_user, 'public', 'CREATE')
                    """
                )
                current_user, can_create = cur.fetchone()
                if not bool(can_create):
                    pytest.skip(
                        f"PostgreSQL user '{current_user}' lacks CREATE privilege on schema public"
                    )

                cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                has_vector = bool(cur.fetchone()[0])
                if not has_vector:
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    except Exception as exc:
                        pytest.skip(
                            "pgvector extension is not installed and could not be created: "
                            f"{exc}"
                        )
    finally:
        conn.close()

    return dsn


@pytest.fixture(scope="session")
def real_embedder(test_config: TestConfig):
    """
    Load a real sentence-transformers embedder once per test session.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        pytest.skip(f"sentence-transformers is unavailable: {exc}")

    try:
        return SentenceTransformer(test_config.embed_model)
    except Exception as exc:
        pytest.skip(f"Failed to load embedding model '{test_config.embed_model}': {exc}")


@pytest.fixture
def real_chromadb_provider(tmp_path: Path):
    """
    Create a real ChromaDB provider against an isolated temp directory.
    """
    from lsm.vectordb.factory import create_vectordb_provider

    config = VectorDBConfig(
        provider="chromadb",
        persist_dir=tmp_path / ".chroma",
        collection=f"test_{tmp_path.name}",
    )
    provider = create_vectordb_provider(config)
    if not provider.is_available():
        pytest.skip("ChromaDB provider is unavailable in this environment")
    return provider


@pytest.fixture
def real_postgresql_provider(live_postgres_connection_string: str):
    """
    Create a real PostgreSQL provider against an isolated collection name.
    """
    from lsm.vectordb.factory import create_vectordb_provider

    collection_name = f"test_pg_{uuid4().hex[:12]}"
    config = VectorDBConfig(
        provider="postgresql",
        connection_string=live_postgres_connection_string,
        collection=collection_name,
        pool_size=2,
    )
    provider = create_vectordb_provider(config)
    if not provider.is_available():
        pytest.skip("PostgreSQL provider is unavailable in this environment")

    try:
        provider.delete_all()
    except Exception:
        pass

    try:
        yield provider
    finally:
        try:
            provider.delete_all()
        except Exception:
            pass
        pool = getattr(provider, "_pool", None)
        if pool is not None:
            try:
                pool.closeall()
            except Exception:
                pass


def _create_live_provider_or_skip(
    *,
    provider_name: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
):
    from lsm.providers.factory import create_provider

    if provider_name != "local" and not api_key:
        pytest.skip(f"Missing credentials for provider '{provider_name}'")
    if provider_name == "local" and not base_url:
        pytest.skip("Set LSM_TEST_OLLAMA_BASE_URL to enable local provider tests")

    config = LLMConfig(
        provider=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    try:
        provider = create_provider(config)
    except Exception as exc:
        pytest.skip(f"Unable to create {provider_name} provider: {exc}")
    if not provider.is_available():
        pytest.skip(f"{provider_name} provider is not available")
    return provider


@pytest.fixture
def real_openai_provider(test_config: TestConfig):
    """Create an OpenAI provider using `LSM_TEST_OPENAI_API_KEY`."""
    return _create_live_provider_or_skip(
        provider_name="openai",
        model="gpt-5.2",
        api_key=test_config.openai_api_key,
    )


@pytest.fixture
def real_anthropic_provider(test_config: TestConfig):
    """Create an Anthropic provider using `LSM_TEST_ANTHROPIC_API_KEY`."""
    return _create_live_provider_or_skip(
        provider_name="anthropic",
        model="claude-sonnet-4-5",
        api_key=test_config.anthropic_api_key,
    )


@pytest.fixture
def real_gemini_provider(test_config: TestConfig):
    """Create a Gemini provider using `LSM_TEST_GOOGLE_API_KEY`."""
    return _create_live_provider_or_skip(
        provider_name="gemini",
        model="gemini-2.5-flash",
        api_key=test_config.google_api_key,
    )


@pytest.fixture
def real_local_provider(test_config: TestConfig):
    """Create a local/Ollama provider using `LSM_TEST_OLLAMA_BASE_URL`."""
    return _create_live_provider_or_skip(
        provider_name="local",
        model="llama3.1",
        base_url=test_config.ollama_base_url,
    )


@pytest.fixture
def rich_test_corpus() -> list[str]:
    """
    Provide realistic, semantically diverse text used in integration tests.
    """
    return [
        (
            "Epistemology studies how we justify belief. In a personal knowledge base, "
            "the key challenge is deciding when a note is evidence versus interpretation."
        ),
        (
            "Retrieval augmented generation combines vector search with language models. "
            "A retriever finds candidate passages, then a synthesizer forms a grounded answer."
        ),
        (
            "ChromaDB stores dense vector embeddings with metadata filters. "
            "Metadata like source_path, language, and tags makes retrieval auditable."
        ),
        (
            "Incremental ingest avoids reprocessing unchanged files by hashing content "
            "and comparing manifest metadata such as modification timestamps."
        ),
        (
            "Citations improve trust in generated answers because readers can inspect "
            "the underlying passages and verify whether claims are supported."
        ),
        (
            "Local-first systems prioritize user control: documents remain on-device, "
            "and network calls are optional rather than required for core workflows."
        ),
    ]


@pytest.fixture
def populated_chromadb(real_chromadb_provider, real_embedder, rich_test_corpus: list[str]):
    """
    Populate a real ChromaDB provider with embeddings for the rich test corpus.
    """
    try:
        vectors = real_embedder.encode(rich_test_corpus, convert_to_numpy=True)
    except Exception as exc:
        pytest.skip(f"Failed to embed rich test corpus: {exc}")

    embeddings = vectors.tolist() if hasattr(vectors, "tolist") else list(vectors)
    ids = [f"corpus_{idx}" for idx in range(len(rich_test_corpus))]
    metadatas = [
        {"source_path": f"/synthetic/doc_{idx}.txt", "chunk_index": idx}
        for idx in range(len(rich_test_corpus))
    ]
    real_chromadb_provider.add_chunks(
        ids=ids,
        documents=rich_test_corpus,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return real_chromadb_provider
