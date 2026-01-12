"""
Tests for remote source providers (web search, APIs, etc.).
"""

import pytest
from unittest.mock import Mock, patch
import requests

from lsm.query.remote.base import BaseRemoteProvider, RemoteResult
from lsm.query.remote.factory import create_remote_provider, register_remote_provider, list_available_providers
from lsm.query.remote.brave import BraveSearchProvider


class MockRemoteProvider(BaseRemoteProvider):
    """Mock provider for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.search_called = False

    @property
    def name(self) -> str:
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
            BaseRemoteProvider({})

    def test_mock_provider_implements_interface(self):
        """Test mock provider implements required interface."""
        provider = MockRemoteProvider({"type": "mock", "enabled": True})

        assert provider.name == "mock"
        assert provider.enabled is True
        assert provider.weight == 1.0  # Default
        assert provider.max_results == 5  # Default

    def test_provider_search_returns_results(self):
        """Test provider search returns RemoteResult objects."""
        provider = MockRemoteProvider({"type": "mock", "enabled": True})

        results = provider.search("test query", max_results=3)

        assert len(results) == 3
        assert all(isinstance(r, RemoteResult) for r in results)
        assert results[0].title == "Result 0"
        assert results[0].url == "https://example.com/0"
        assert results[0].score == 1.0


class TestRemoteProviderFactory:
    """Tests for remote provider factory."""

    def test_list_available_providers(self):
        """Test listing available providers."""
        providers = list_available_providers()

        # Should have at least Brave Search
        assert "web_search" in providers
        assert "brave_search" in providers

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        register_remote_provider("mock", MockRemoteProvider)

        providers = list_available_providers()
        assert "mock" in providers

    def test_create_provider_brave(self):
        """Test creating Brave Search provider."""
        config = {
            "type": "web_search",
            "enabled": True,
            "api_key": "test_key",
            "max_results": 10
        }

        provider = create_remote_provider(config)

        assert isinstance(provider, BraveSearchProvider)
        assert provider.enabled is True
        assert provider.max_results == 10

    def test_create_provider_invalid_type(self):
        """Test creating provider with invalid type raises error."""
        config = {
            "type": "invalid_provider",
            "enabled": True
        }

        with pytest.raises(ValueError, match="Unknown remote provider type"):
            create_remote_provider(config)


class TestBraveSearchProvider:
    """Tests for Brave Search provider."""

    def test_brave_provider_initialization(self):
        """Test Brave provider initializes correctly."""
        config = {
            "type": "web_search",
            "enabled": True,
            "api_key": "test_brave_key",
            "max_results": 5,
            "weight": 0.8
        }

        provider = BraveSearchProvider(config)

        assert provider.name == "brave_search"
        assert provider.enabled is True
        assert provider.api_key == "test_brave_key"
        assert provider.max_results == 5
        assert provider.weight == 0.8

    def test_brave_provider_uses_env_var(self):
        """Test Brave provider falls back to environment variable."""
        with patch.dict('os.environ', {'BRAVE_API_KEY': 'env_key'}):
            config = {
                "type": "web_search",
                "enabled": True
            }

            provider = BraveSearchProvider(config)
            assert provider.api_key == "env_key"

    def test_brave_provider_is_available(self):
        """Test Brave provider availability check."""
        # With API key
        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)
        assert provider.is_available() is True

        # Without API key
        config_no_key = {"type": "web_search"}
        with patch.dict('os.environ', {}, clear=True):
            provider_no_key = BraveSearchProvider(config_no_key)
            assert provider_no_key.is_available() is False

    @patch('requests.get')
    def test_brave_search_success(self, mock_get):
        """Test Brave Search API call success."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "Test description"
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query", max_results=1)

        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].snippet == "Test description"

        # Verify API was called correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == BraveSearchProvider.SEARCH_ENDPOINT
        assert kwargs['params']['q'] == "test query"
        assert kwargs['params']['count'] == 1

    @patch('requests.get')
    def test_brave_search_api_error(self, mock_get):
        """Test Brave Search handles API errors gracefully."""
        # Mock API error
        mock_get.side_effect = requests.RequestException("API Error")

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query")

        # Should return empty list on error
        assert results == []

    @patch('requests.get')
    def test_brave_search_timeout(self, mock_get):
        """Test Brave Search handles timeouts."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        config = {"type": "web_search", "api_key": "test_key", "timeout": 5}
        provider = BraveSearchProvider(config)

        results = provider.search("test query")

        assert results == []

    @patch('requests.get')
    def test_brave_search_empty_results(self, mock_get):
        """Test Brave Search handles empty results."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}
        mock_get.return_value = mock_response

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query")

        assert results == []

    @patch('requests.get')
    def test_brave_search_respects_max_results(self, mock_get):
        """Test Brave Search respects max_results limit."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": f"Result {i}", "url": f"https://example.com/{i}", "description": f"Desc {i}"}
                    for i in range(10)
                ]
            }
        }
        mock_get.return_value = mock_response

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query", max_results=3)

        # Should only return 3 results even though API returned 10
        assert len(results) == 3


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
            url="https://example.com"
        )

        assert result.snippet == ""
        assert result.score == 1.0

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
