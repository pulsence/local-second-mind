"""
Tests for NYTimes provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.news.nytimes import NYTimesProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


NYT_ARTICLE_RESPONSE = {
    "response": {
        "docs": [
            {
                "_id": "nyt-1",
                "web_url": "https://nytimes.com/1",
                "headline": {"main": "NYT Article"},
                "snippet": "Article snippet.",
                "pub_date": "2024-01-01T00:00:00Z",
                "section_name": "World",
                "byline": {"original": "By Reporter"},
            }
        ]
    }
}

NYT_TOP_RESPONSE = {
    "results": [
        {
            "title": "Top Story",
            "url": "https://nytimes.com/top",
            "abstract": "Top story summary",
            "published_date": "2024-01-02T00:00:00Z",
            "section": "US",
            "byline": "By Reporter",
            "short_url": "https://nyti.ms/1",
        }
    ]
}


class TestNYTimesProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = NYTimesProvider({})
        with pytest.raises(ValueError, match="NYTimes requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_article_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = NYT_ARTICLE_RESPONSE
        mock_get.return_value = response

        provider = NYTimesProvider({"api_key": "test-key"})
        results = provider.search("election", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["nyt_id"] == "nyt-1"
        self.assert_valid_output(results)

    @patch("requests.get")
    def test_top_stories_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = NYT_TOP_RESPONSE
        mock_get.return_value = response

        provider = NYTimesProvider({"api_key": "test-key", "top_stories_section": "home"})
        results = provider.search("", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["nyt_id"] == "https://nyti.ms/1"
        self.assert_valid_output(results)
