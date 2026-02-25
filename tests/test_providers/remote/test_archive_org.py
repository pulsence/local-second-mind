"""
Tests for Archive.org provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.cultural.archive_org import ArchiveOrgProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


ARCHIVE_RESPONSE = {
    "response": {
        "docs": [
            {
                "identifier": "item1",
                "title": "Archive Item One",
                "description": "Historical archive item.",
                "creator": ["Alice"],
                "year": 1920,
                "mediatype": "texts",
                "collection": ["opensource"],
                "downloads": 42,
                "subject": ["history"],
            }
        ]
    }
}


class TestArchiveOrgProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = ARCHIVE_RESPONSE
        mock_get.return_value = response

        provider = ArchiveOrgProvider({})
        results = provider.search("archive", max_results=1)

        assert len(results) == 1
        assert results[0].metadata["identifier"] == "item1"
        assert results[0].metadata["mediatype"] == "texts"
        self.assert_valid_output(results)
