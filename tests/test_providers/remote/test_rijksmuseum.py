"""
Tests for Rijksmuseum provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.cultural.rijksmuseum import RijksmuseumProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


RIJKSMUSEUM_SEARCH_RESPONSE = {
    "artObjects": [
        {
            "objectNumber": "SK-C-5",
            "title": "The Night Watch",
            "principalOrFirstMaker": "Rembrandt",
            "links": {"web": "https://www.rijksmuseum.nl/en/collection/SK-C-5"},
            "webImage": {"url": "https://example.org/nightwatch.jpg"},
        }
    ]
}

RIJKSMUSEUM_DETAIL_RESPONSE = {
    "artObject": {
        "objectNumber": "SK-C-5",
        "title": "The Night Watch",
        "principalOrFirstMaker": "Rembrandt",
        "links": {"web": "https://www.rijksmuseum.nl/en/collection/SK-C-5"},
        "webImage": {"url": "https://example.org/nightwatch.jpg"},
        "dating": {"year": 1642},
    }
}


class TestRijksmuseumProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = RijksmuseumProvider({})
        with pytest.raises(ValueError, match="Rijksmuseum requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = RIJKSMUSEUM_SEARCH_RESPONSE

        detail_response = Mock()
        detail_response.status_code = 200
        detail_response.json.return_value = RIJKSMUSEUM_DETAIL_RESPONSE

        mock_get.side_effect = [search_response, detail_response]

        provider = RijksmuseumProvider({"api_key": "test-key"})
        results = provider.search("night watch", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["object_number"] == "SK-C-5"
        self.assert_valid_output(results)
