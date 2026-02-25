"""
Tests for Smithsonian provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.cultural.smithsonian import SmithsonianProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


SMITHSONIAN_RESPONSE = {
    "response": {
        "rows": [
            {
                "id": "row1",
                "content": {
                    "title": "Smithsonian Artifact",
                    "descriptiveNonRepeating": {
                        "record_ID": "smith-1",
                        "guid": "https://example.org/smith-1",
                        "date": "1910",
                    },
                },
            }
        ]
    }
}


class TestSmithsonianProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = SmithsonianProvider({})
        with pytest.raises(ValueError, match="Smithsonian requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = SMITHSONIAN_RESPONSE
        mock_get.return_value = response

        provider = SmithsonianProvider({"api_key": "test-key"})
        results = provider.search("artifact", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["smithsonian_id"] == "smith-1"
        self.assert_valid_output(results)
