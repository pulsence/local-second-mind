"""
Tests for IIIF provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.cultural.iiif import IIIFProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


IIIF_RESPONSE = {
    "resources": [
        {
            "@id": "https://example.org/iiif/resource/1",
            "label": "Page 1",
            "chars": "Sample transcription",
            "within": "https://example.org/iiif/manifest/1",
        }
    ]
}


class TestIIIFProvider(RemoteProviderOutputTest):
    def test_validate_requires_endpoint(self):
        provider = IIIFProvider({})
        with pytest.raises(ValueError, match="IIIF provider requires an endpoint"):
            provider.validate_config()

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = IIIF_RESPONSE
        mock_get.return_value = response

        provider = IIIFProvider({"endpoint": "https://example.org/iiif/search"})
        results = provider.search("manuscript", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["iiif_id"] == "https://example.org/iiif/resource/1"
        self.assert_valid_output(results)
