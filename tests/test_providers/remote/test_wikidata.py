"""
Tests for Wikidata provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.cultural.wikidata import WikidataProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


WIKIDATA_RESPONSE = {
    "results": {
        "bindings": [
            {
                "item": {"value": "http://www.wikidata.org/entity/Q1"},
                "itemLabel": {"value": "Universe"},
                "description": {"value": "totality of space and time"},
            }
        ]
    }
}


class TestWikidataProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = WIKIDATA_RESPONSE
        mock_get.return_value = response

        provider = WikidataProvider({})
        results = provider.search("universe", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["wikidata_id"] == "Q1"
        self.assert_valid_output(results)
