"""
Tests for DPLA provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.cultural.dpla import DPLAProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


DPLA_RESPONSE = {
    "docs": [
        {
            "id": "dpla-1",
            "isShownAt": "https://example.org/item",
            "sourceResource": {
                "title": "DPLA Item",
                "description": "Library record",
                "creator": ["Archivist"],
                "date": "1890",
                "subject": [{"name": "History"}],
            },
            "provider": {"name": "Sample Library"},
        }
    ]
}


class TestDPLAProvider(RemoteProviderOutputTest):
    def test_validate_requires_api_key(self):
        provider = DPLAProvider({})
        with pytest.raises(ValueError, match="DPLA requires an API key"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = DPLA_RESPONSE
        mock_get.return_value = response

        provider = DPLAProvider({"api_key": "test-key"})
        results = provider.search("history", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["dpla_id"] == "dpla-1"
        assert results[0].metadata["provider"] == "Sample Library"
        self.assert_valid_output(results)
