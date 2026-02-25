"""
Tests for Unpaywall provider implementation.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.providers.academic.unpaywall import UnpaywallProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


UNPAYWALL_RESPONSE = {
    "doi": "10.1038/s41586-019-1099-1",
    "doi_url": "https://doi.org/10.1038/s41586-019-1099-1",
    "title": "Deep Learning Methods for Natural Language Processing",
    "is_oa": True,
    "oa_status": "gold",
    "best_oa_location": {
        "url": "https://example.org/open-access",
        "url_for_pdf": "https://example.org/open-access.pdf",
        "license": "cc-by",
    },
    "oa_locations": [
        {"url": "https://example.org/open-access"},
    ],
    "journal_name": "Nature",
    "publisher": "Nature Publishing Group",
    "published_date": "2023-06-15",
}


class TestUnpaywallProvider(RemoteProviderOutputTest):
    def test_provider_initialization_defaults(self):
        provider = UnpaywallProvider({"email": "test@example.com"})

        assert provider.endpoint == "https://api.unpaywall.org/v2"
        assert provider.min_interval_seconds == 0.5
        assert provider.snippet_max_chars == 600
        assert provider.timeout == 15

    def test_validate_requires_email(self):
        provider = UnpaywallProvider({})
        with pytest.raises(ValueError, match="Unpaywall requires a contact email"):
            provider.validate_config()

    def test_get_name(self):
        provider = UnpaywallProvider({"email": "test@example.com"})
        assert provider.get_name() == "Unpaywall"
        assert provider.name == "unpaywall"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = UNPAYWALL_RESPONSE
        mock_get.return_value = response

        provider = UnpaywallProvider({"email": "test@example.com"})
        results = provider.search("10.1038/s41586-019-1099-1", max_results=1)

        assert len(results) == 1
        assert results[0].url == "https://example.org/open-access"
        assert results[0].metadata["doi"] == "10.1038/s41586-019-1099-1"
        assert results[0].metadata["pdf_url"] == "https://example.org/open-access.pdf"
        assert results[0].metadata["license"] == "cc-by"
        assert results[0].metadata["source_id"] == "10.1038/s41586-019-1099-1"

        self.assert_valid_output(results)

    @patch("requests.get")
    def test_search_structured_prefers_doi_field(self, mock_get):
        response = Mock()
        response.status_code = 200
        response.json.return_value = UNPAYWALL_RESPONSE
        mock_get.return_value = response

        provider = UnpaywallProvider({"email": "test@example.com"})
        results = provider.search_structured({"doi": "10.1038/s41586-019-1099-1"})

        assert len(results) == 1
        assert results[0]["doi"] == "10.1038/s41586-019-1099-1"

