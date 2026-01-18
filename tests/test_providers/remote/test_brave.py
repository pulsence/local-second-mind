"""
Tests for Brave Search provider implementation.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from lsm.remote.providers.brave import BraveSearchProvider


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
        assert provider.endpoint == BraveSearchProvider.SEARCH_ENDPOINT

    def test_brave_provider_endpoint_defaults_when_none(self):
        """Test endpoint defaults when config provides None."""
        config = {
            "type": "web_search",
            "enabled": True,
            "api_key": "test_brave_key",
            "endpoint": None,
        }

        provider = BraveSearchProvider(config)
        assert provider.endpoint == BraveSearchProvider.SEARCH_ENDPOINT

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

    @patch('requests.get')
    def test_brave_search_malformed_response(self, mock_get):
        """Test Brave Search handles malformed API responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}
        mock_get.return_value = mock_response

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query")

        # Should return empty list on malformed response
        assert results == []

    @patch('requests.get')
    def test_brave_search_http_error(self, mock_get):
        """Test Brave Search handles HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 429  # Rate limit
        mock_response.raise_for_status.side_effect = requests.HTTPError("Rate limited")
        mock_get.return_value = mock_response

        config = {"type": "web_search", "api_key": "test_key"}
        provider = BraveSearchProvider(config)

        results = provider.search("test query")

        # Should return empty list on HTTP error
        assert results == []
