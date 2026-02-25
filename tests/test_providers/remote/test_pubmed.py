"""
Tests for PubMed provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.academic.pubmed import PubMedProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


PUBMED_SEARCH_RESPONSE = {
    "esearchresult": {
        "idlist": ["123456", "7891011"],
    }
}

PUBMED_SUMMARY_RESPONSE = {
    "result": {
        "uids": ["123456", "7891011"],
        "123456": {
            "uid": "123456",
            "title": "Neural Language Models in Healthcare",
            "authors": [{"name": "Jane Doe"}, {"name": "John Smith"}],
            "pubdate": "2023 Jun 15",
            "fulljournalname": "Journal of Medical AI",
            "articleids": [
                {"idtype": "doi", "value": "10.1234/medical.ai"},
                {"idtype": "pmcid", "value": "PMC12345"},
            ],
            "elocationid": "doi:10.1234/medical.ai",
        },
        "7891011": {
            "uid": "7891011",
            "title": "Biomedical Retrieval Systems",
            "authors": [{"name": "Alice Johnson"}],
            "pubdate": "2022 Oct",
            "source": "BioSearch",
            "articleids": [
                {"idtype": "doi", "value": "10.5678/bio.search"},
            ],
        },
    }
}


class TestPubMedProvider(RemoteProviderOutputTest):
    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = PUBMED_SEARCH_RESPONSE

        summary_response = Mock()
        summary_response.status_code = 200
        summary_response.json.return_value = PUBMED_SUMMARY_RESPONSE

        mock_get.side_effect = [search_response, summary_response]

        provider = PubMedProvider({})
        results = provider.search("neural language models", max_results=2)

        assert len(results) == 2
        assert results[0].metadata["pmid"] == "123456"
        assert results[0].metadata["doi"] == "10.1234/medical.ai"
        assert results[0].metadata["pmcid"] == "PMC12345"
        assert results[0].metadata["pmc_url"].endswith("/PMC12345/")
        assert results[0].metadata["pdf_url"].endswith("/PMC12345/pdf/")
        assert results[0].metadata["source_id"] == "10.1234/medical.ai"

        self.assert_valid_output(results)

    def test_get_name(self):
        provider = PubMedProvider({})
        assert provider.get_name() == "PubMed"
        assert provider.name == "pubmed"
