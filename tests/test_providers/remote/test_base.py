"""
Tests for BaseRemoteProvider abstract interface.
"""

import pytest

from lsm.remote.base import BaseRemoteProvider, RemoteResult


class MockRemoteProvider(BaseRemoteProvider):
    """Mock provider for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.search_called = False

    def get_name(self) -> str:
        return "mock"

    def search(self, query: str, max_results: int = 5):
        self.search_called = True
        return [
            RemoteResult(
                title=f"Result {i}",
                url=f"https://example.com/{i}",
                snippet=f"Snippet for result {i}",
                score=1.0 - (i * 0.1)
            )
            for i in range(max_results)
        ]


class TestRemoteProviderBase:
    """Tests for BaseRemoteProvider interface."""

    def test_base_provider_requires_implementation(self):
        """Test that BaseRemoteProvider requires implementing search()."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseRemoteProvider({})  # type: ignore

    def test_mock_provider_implements_interface(self):
        """Test mock provider implements required interface."""
        provider = MockRemoteProvider({"type": "mock"})

        assert provider.get_name() == "mock"
        assert provider.config["type"] == "mock"

    def test_provider_search_returns_results(self):
        """Test provider search returns RemoteResult objects."""
        provider = MockRemoteProvider({"type": "mock"})

        results = provider.search("test query", max_results=3)

        assert len(results) == 3
        assert all(isinstance(r, RemoteResult) for r in results)
        assert results[0].title == "Result 0"
        assert results[0].url == "https://example.com/0"
        assert results[0].score == 1.0

    def test_provider_validate_config(self):
        """Test provider configuration validation."""
        provider = MockRemoteProvider({"type": "mock"})

        # Default implementation doesn't raise
        provider.validate_config()  # Should not raise


class TestRemoteResult:
    """Tests for RemoteResult dataclass."""

    def test_remote_result_creation(self):
        """Test creating RemoteResult."""
        result = RemoteResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            score=0.95
        )

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.score == 0.95

    def test_remote_result_defaults(self):
        """Test RemoteResult with default values."""
        result = RemoteResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet"  # Required field
        )

        assert result.score == 1.0
        assert result.metadata == {}

    def test_remote_result_to_dict(self):
        """Test converting RemoteResult to dict."""
        result = RemoteResult(
            title="Test",
            url="https://example.com",
            snippet="Snippet",
            score=0.9
        )

        as_dict = {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "score": result.score
        }

        assert as_dict["title"] == "Test"
        assert as_dict["url"] == "https://example.com"
        assert as_dict["snippet"] == "Snippet"
        assert as_dict["score"] == 0.9
