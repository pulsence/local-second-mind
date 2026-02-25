"""
Tests for Met provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.cultural.met import MetProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


MET_SEARCH_RESPONSE = {"objectIDs": [1]}

MET_OBJECT_RESPONSE = {
    "objectID": 1,
    "title": "Ancient Vase",
    "objectURL": "https://www.metmuseum.org/art/collection/search/1",
    "artistDisplayName": "Unknown",
    "objectEndDate": 1900,
    "primaryImageSmall": "https://example.org/image.jpg",
}


class TestMetProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = MET_SEARCH_RESPONSE

        object_response = Mock()
        object_response.status_code = 200
        object_response.json.return_value = MET_OBJECT_RESPONSE

        mock_get.side_effect = [search_response, object_response]

        provider = MetProvider({})
        results = provider.search("vase", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["met_id"] == 1
        self.assert_valid_output(results)
