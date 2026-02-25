"""
Tests for Guardian provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.news.guardian import GuardianProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


GUARDIAN_RESPONSE = {
    "response": {
        "results": [
            {
                "id": "world/2024/jan/01/example",
                "webTitle": "Guardian Article",
                "webUrl": "https://www.theguardian.com/world/2024/jan/01/example",
                "webPublicationDate": "2024-01-01T00:00:00Z",
                "sectionName": "World",
                "fields": {"trailText": "Summary"},
            }
        ]
    }
}


class TestGuardianProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = GuardianProvider({})
        with pytest.raises(ValueError, match="Guardian requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = GUARDIAN_RESPONSE
        mock_get.return_value = response

        provider = GuardianProvider({"api_key": "test-key"})
        results = provider.search("climate", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["guardian_id"] == "world/2024/jan/01/example"
        self.assert_valid_output(results)
