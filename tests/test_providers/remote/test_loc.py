"""
Tests for Library of Congress provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.cultural.loc import LOCProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


LOC_RESPONSE = {
    "results": [
        {
            "id": "loc:123",
            "title": "LOC Item",
            "url": "https://loc.gov/item/123",
            "description": "Library of Congress record.",
            "subject": ["archives"],
            "date": "1901",
        }
    ]
}


class TestLOCProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = LOC_RESPONSE
        mock_get.return_value = response

        provider = LOCProvider({})
        results = provider.search("archives", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["loc_id"] == "loc:123"
        self.assert_valid_output(results)
