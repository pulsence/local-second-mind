"""
Tests for NewsAPI provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.news.newsapi import NewsAPIProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


NEWSAPI_RESPONSE = {
    "articles": [
        {
            "title": "NewsAPI Article",
            "url": "https://example.org/newsapi",
            "description": "NewsAPI summary",
            "publishedAt": "2024-01-01T00:00:00Z",
            "source": {"name": "Example"},
            "author": "Reporter",
        }
    ]
}


class TestNewsAPIProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = NewsAPIProvider({})
        with pytest.raises(ValueError, match="NewsAPI requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = NEWSAPI_RESPONSE
        mock_get.return_value = response

        provider = NewsAPIProvider({"api_key": "test-key"})
        results = provider.search("economy", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["source"] == "Example"
        self.assert_valid_output(results)
