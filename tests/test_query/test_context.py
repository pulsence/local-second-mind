"""
Tests for query context module.
"""

import pytest
import time
from pathlib import Path

from lsm.query.context import (
    build_context_block,
    fallback_answer,
    fetch_remote_sources,
)
from lsm.ui.utils import format_source_list
from lsm.query.session import Candidate
from lsm.config.models import (
    LSMConfig,
    IngestConfig,
    QueryConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    LLMServiceConfig,
    VectorDBConfig,
    RemoteProviderConfig,
)


class TestBuildContextBlock:
    """Tests for context block building."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates."""
        return [
            Candidate(
                cid="chunk1",
                text="Python is a high-level programming language.",
                meta={
                    "source_path": "/docs/python.md",
                    "source_name": "python.md",
                    "chunk_index": 0,
                    "ext": ".md",
                },
                distance=0.1,
            ),
            Candidate(
                cid="chunk2",
                text="Python has extensive standard library.",
                meta={
                    "source_path": "/docs/python.md",
                    "source_name": "python.md",
                    "chunk_index": 1,
                    "ext": ".md",
                },
                distance=0.15,
            ),
        ]

    def test_build_context_block_basic(self, sample_candidates):
        """Test building basic context block."""
        context, sources = build_context_block(sample_candidates)

        # Check context contains source labels
        assert "[S1]" in context
        assert "[S2]" in context

        # Check context contains text
        assert "Python is a high-level" in context
        assert "extensive standard library" in context

        # Check sources list
        assert len(sources) == 2
        assert sources[0]["label"] == "S1"
        assert sources[1]["label"] == "S2"

    def test_build_context_block_includes_metadata(self, sample_candidates):
        """Test that context includes metadata."""
        context, sources = build_context_block(sample_candidates)

        # Check metadata in context
        assert "python.md" in context
        assert "chunk_index=0" in context
        assert "chunk_index=1" in context

        # Check metadata in sources
        assert sources[0]["source_path"] == "/docs/python.md"
        assert sources[0]["chunk_index"] == 0
        assert sources[1]["chunk_index"] == 1

    def test_build_context_block_empty_list(self):
        """Test building context from empty list."""
        context, sources = build_context_block([])

        assert context == ""
        assert sources == []

    def test_build_context_block_minimal_metadata(self):
        """Test building context with minimal metadata."""
        candidates = [
            Candidate(
                cid="1",
                text="Some text",
                meta={},  # Minimal metadata
                distance=0.1,
            )
        ]

        context, sources = build_context_block(candidates)

        assert "[S1]" in context
        assert "Some text" in context
        assert sources[0]["source_path"] == "unknown"

    def test_build_context_block_all_metadata_fields(self):
        """Test context with all metadata fields."""
        candidates = [
            Candidate(
                cid="1",
                text="Full metadata",
                meta={
                    "source_path": "/docs/test.md",
                    "source_name": "test.md",
                    "chunk_index": 5,
                    "ext": ".md",
                    "mtime_ns": 1234567890,
                    "file_hash": "abc123",
                    "ingested_at": "2026-01-10",
                },
                distance=0.2,
            )
        ]

        context, sources = build_context_block(candidates)

        # Check all fields present in context
        assert "mtime_ns=1234567890" in context
        assert "file_hash=abc123" in context
        assert "ingested_at=2026-01-10" in context

        # Check all fields in sources
        assert sources[0]["mtime_ns"] == 1234567890
        assert sources[0]["file_hash"] == "abc123"
        assert sources[0]["ingested_at"] == "2026-01-10"


class TestFallbackAnswer:
    """Tests for fallback answer generation."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates."""
        return [
            Candidate(
                cid="1",
                text="Python is a programming language.",
                meta={"source_path": "/docs/python.md", "chunk_index": 0},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Python supports multiple paradigms.",
                meta={"source_path": "/docs/python.md", "chunk_index": 1},
                distance=0.15,
            ),
        ]

    def test_fallback_answer_basic(self, sample_candidates):
        """Test basic fallback answer generation."""
        answer = fallback_answer("What is Python?", sample_candidates)

        # Check warning message
        assert "unavailable" in answer.lower()

        # Check question is included
        assert "What is Python?" in answer

        # Check excerpts are included
        assert "[S1]" in answer
        assert "[S2]" in answer
        assert "Python is a programming language" in answer

    def test_fallback_answer_truncates_long_text(self):
        """Test that fallback truncates long excerpts."""
        long_text = "x" * 2000
        candidates = [
            Candidate(
                cid="1",
                text=long_text,
                meta={"source_path": "/docs/test.md", "chunk_index": 0},
                distance=0.1,
            )
        ]

        answer = fallback_answer("Question?", candidates, max_chars=500)

        assert "[truncated]" in answer
        assert len(answer) < 1000  # Should be significantly shorter

    def test_fallback_answer_empty_candidates(self):
        """Test fallback with no candidates."""
        answer = fallback_answer("Question?", [])

        assert "unavailable" in answer.lower()
        assert "Question?" in answer

    def test_fallback_answer_includes_metadata(self, sample_candidates):
        """Test that fallback includes source metadata."""
        answer = fallback_answer("Question?", sample_candidates)

        assert "/docs/python.md" in answer
        assert "chunk_index=0" in answer
        assert "chunk_index=1" in answer


class TestFormatSourceList:
    """Tests for source list formatting."""

    def test_format_source_list_basic(self):
        """Test basic source list formatting."""
        sources = [
            {
                "label": "S1",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
            {
                "label": "S2",
                "source_path": "/docs/java.md",
                "source_name": "java.md",
            },
        ]

        formatted = format_source_list(sources)

        assert "Sources:" in formatted
        assert "[S1]" in formatted
        assert "[S2]" in formatted
        assert "python.md" in formatted
        assert "java.md" in formatted
        assert "/docs/python.md" in formatted

    def test_format_source_list_groups_by_file(self):
        """Test that sources from same file are grouped."""
        sources = [
            {
                "label": "S1",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
            {
                "label": "S2",
                "source_path": "/docs/java.md",
                "source_name": "java.md",
            },
            {
                "label": "S3",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
        ]

        formatted = format_source_list(sources)

        # Should group S1 and S3 together
        assert "[S1] [S3]" in formatted or "[S1]" in formatted and "[S3]" in formatted
        lines = formatted.split("\n")
        # Should have 3 lines: header, python.md, java.md
        assert len([l for l in lines if l.strip() and l.startswith("- ")]) == 2

    def test_format_source_list_empty(self):
        """Test formatting empty source list."""
        formatted = format_source_list([])
        assert formatted == ""

    def test_format_source_list_missing_name(self):
        """Test formatting with missing source_name."""
        sources = [
            {
                "label": "S1",
                "source_path": "/docs/test.md",
                # source_name missing
            }
        ]

        formatted = format_source_list(sources)

        # Should derive name from path
        assert "test.md" in formatted or "/docs/test.md" in formatted

    def test_format_source_list_preserves_order(self):
        """Test that source order is preserved."""
        sources = [
            {
                "label": "S1",
                "source_path": "/docs/a.md",
                "source_name": "a.md",
            },
            {
                "label": "S2",
                "source_path": "/docs/b.md",
                "source_name": "b.md",
            },
            {
                "label": "S3",
                "source_path": "/docs/c.md",
                "source_name": "c.md",
            },
        ]

        formatted = format_source_list(sources)
        lines = [l for l in formatted.split("\n") if l.strip().startswith("- ")]

        # First appearance of each file should be in order
        assert "a.md" in lines[0]
        assert "b.md" in lines[1]
        assert "c.md" in lines[2]

    def test_format_source_list_multiple_labels_per_file(self):
        """Test formatting when multiple chunks from same file."""
        sources = [
            {
                "label": "S1",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
            {
                "label": "S2",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
            {
                "label": "S3",
                "source_path": "/docs/python.md",
                "source_name": "python.md",
            },
        ]

        formatted = format_source_list(sources)

        # All labels should appear together
        assert "[S1]" in formatted
        assert "[S2]" in formatted
        assert "[S3]" in formatted

        # Should only have one line for python.md
        lines = [l for l in formatted.split("\n") if "python.md" in l]
        assert len(lines) == 1

    def test_format_source_list_handles_unknown_path(self):
        """Test formatting with unknown source path."""
        sources = [
            {
                "label": "S1",
                # No source_path
                "source_name": None,
            }
        ]

        formatted = format_source_list(sources)

        assert "[S1]" in formatted
        assert "unknown" in formatted.lower()


def test_fetch_remote_sources_timeout_is_handled(monkeypatch):
    """Slow remote provider should time out and return no results."""

    class SlowProvider:
        def search(self, question, max_results=5):
            time.sleep(1.2)
            return []

    def fake_create_remote_provider(_type, _cfg):
        return SlowProvider()

    monkeypatch.setattr("lsm.query.context.create_remote_provider", fake_create_remote_provider)

    config = LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="hybrid"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(path=Path("/tmp/.chroma"), collection="test"),
        remote_providers=[
            RemoteProviderConfig(name="slow", type="fake", timeout=1),
        ],
        config_path=Path("/tmp/config.json"),
    )
    mode_config = config.get_mode_config("hybrid")
    mode_config.source_policy.remote.enabled = True
    mode_config.source_policy.remote.remote_providers = ["slow"]

    results = fetch_remote_sources("test", config, mode_config)
    assert results == []


def test_fetch_remote_sources_uses_provider_cache_when_enabled(monkeypatch):
    """Provider cache hit should avoid live provider calls."""

    def _cached_results(**_kwargs):
        return [
            {
                "title": "Cached",
                "url": "https://cached.example",
                "snippet": "from cache",
                "score": 0.8,
                "metadata": {"cached": True},
            }
        ]

    monkeypatch.setattr("lsm.query.context.load_cached_results", _cached_results)
    monkeypatch.setattr(
        "lsm.query.context.create_remote_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not call provider")),
    )

    config = LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="hybrid"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(path=Path("/tmp/.chroma"), collection="test"),
        remote_providers=[
            RemoteProviderConfig(
                name="wiki",
                type="fake",
                weight=0.5,
                cache_results=True,
                cache_ttl=3600,
            ),
        ],
        config_path=Path("/tmp/config.json"),
    )
    mode_config = config.get_mode_config("hybrid")
    mode_config.source_policy.remote.enabled = True
    mode_config.source_policy.remote.remote_providers = ["wiki"]

    results = fetch_remote_sources("test", config, mode_config)
    assert len(results) == 1
    assert results[0]["title"] == "Cached"
    assert results[0]["weighted_score"] == pytest.approx(0.4)


def test_fetch_remote_sources_saves_cache_on_miss(monkeypatch):
    """Provider cache miss should save fetched results."""

    class _Provider:
        def search(self, _question, max_results=5):
            class _Res:
                title = "Live"
                url = "https://live.example"
                snippet = "live result"
                score = 0.7
                metadata = {"live": True}

            return [_Res()]

    saved_calls = []

    monkeypatch.setattr("lsm.query.context.load_cached_results", lambda **_kwargs: None)
    monkeypatch.setattr("lsm.query.context.create_remote_provider", lambda *_args, **_kwargs: _Provider())
    monkeypatch.setattr(
        "lsm.query.context.save_results",
        lambda **kwargs: saved_calls.append(kwargs),
    )

    config = LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="hybrid"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(path=Path("/tmp/.chroma"), collection="test"),
        remote_providers=[
            RemoteProviderConfig(
                name="wiki",
                type="fake",
                weight=1.0,
                cache_results=True,
                cache_ttl=3600,
            ),
        ],
        config_path=Path("/tmp/config.json"),
    )
    mode_config = config.get_mode_config("hybrid")
    mode_config.source_policy.remote.enabled = True
    mode_config.source_policy.remote.remote_providers = ["wiki"]

    results = fetch_remote_sources("test", config, mode_config)
    assert len(results) == 1
    assert saved_calls
    assert saved_calls[0]["provider_name"] == "wiki"


def test_fetch_remote_sources_passes_provider_specific_options(monkeypatch):
    captured_cfg = {}

    class _Provider:
        def search(self, _question, max_results=5):
            class _Res:
                title = "Live"
                url = "https://live.example"
                snippet = "live result"
                score = 0.7
                metadata = {}

            return [_Res()]

    def _factory(_provider_type, cfg):
        captured_cfg.update(cfg)
        return _Provider()

    monkeypatch.setattr("lsm.query.context.load_cached_results", lambda **_kwargs: None)
    monkeypatch.setattr("lsm.query.context.create_remote_provider", _factory)

    config = LSMConfig(
        ingest=IngestConfig(roots=[Path("/tmp")], manifest=Path("/tmp/manifest.json")),
        query=QueryConfig(mode="hybrid"),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="openai", api_key="test")],
            services={"query": LLMServiceConfig(provider="openai", model="gpt-5.2")},
        ),
        vectordb=VectorDBConfig(path=Path("/tmp/.chroma"), collection="test"),
        remote_providers=[
            RemoteProviderConfig(
                name="openalex",
                type="openalex",
                extra={"email": "you@example.com", "year_from": 2020},
            ),
        ],
        config_path=Path("/tmp/config.json"),
    )
    mode_config = config.get_mode_config("hybrid")
    mode_config.source_policy.remote.enabled = True
    mode_config.source_policy.remote.remote_providers = ["openalex"]

    results = fetch_remote_sources("test", config, mode_config)
    assert len(results) == 1
    assert captured_cfg["email"] == "you@example.com"
    assert captured_cfg["year_from"] == 2020
