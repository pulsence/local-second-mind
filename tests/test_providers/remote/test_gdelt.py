"""
Tests for GDELT provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.news.gdelt import GDELTProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


GDELT_RESPONSE = {
    "articles": [
        {
            "title": "Global Event",
            "url": "https://example.org/news",
            "seendate": "20240101000000",
            "domain": "example.org",
            "language": "en",
        }
    ]
}


class TestGDELTProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = GDELT_RESPONSE
        mock_get.return_value = response

        provider = GDELTProvider({})
        results = provider.search("global event", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["source"] == "example.org"
        self.assert_valid_output(results)
