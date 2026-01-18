"""
Tests for Crossref provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.remote.providers.crossref import CrossrefProvider


CROSSREF_RESPONSE = {
    "status": "ok",
    "message-type": "work-list",
    "message-version": "1.0.0",
    "message": {
        "total-results": 2,
        "items": [
            {
                "DOI": "10.1038/s41586-019-1099-1",
                "URL": "https://doi.org/10.1038/s41586-019-1099-1",
                "title": ["Deep Learning Methods for Natural Language Processing"],
                "abstract": "<p>This paper presents a comprehensive review of deep learning techniques applied to NLP tasks.</p>",
                "author": [
                    {"given": "John", "family": "Smith", "ORCID": "https://orcid.org/0000-0001-2345-6789"},
                    {"given": "Jane", "family": "Doe"},
                ],
                "published": {
                    "date-parts": [[2023, 6, 15]],
                },
                "container-title": ["Nature"],
                "publisher": "Nature Publishing Group",
                "type": "journal-article",
                "is-referenced-by-count": 500,
                "references-count": 45,
                "ISSN": ["0028-0836", "1476-4687"],
                "subject": ["Multidisciplinary"],
                "link": [
                    {
                        "URL": "https://www.nature.com/articles/s41586-019-1099-1.pdf",
                        "content-type": "application/pdf",
                    }
                ],
                "license": [
                    {"URL": "https://creativecommons.org/licenses/by/4.0/"},
                ],
            },
            {
                "DOI": "10.1234/example.2",
                "URL": "https://doi.org/10.1234/example.2",
                "title": ["Transformer Models: A Survey"],
                "abstract": None,
                "author": [
                    {"given": "Alice", "family": "Johnson"},
                ],
                "published-print": {
                    "date-parts": [[2022, 10]],
                },
                "container-title": ["ACL Anthology"],
                "publisher": "Association for Computational Linguistics",
                "type": "proceedings-article",
                "is-referenced-by-count": 100,
                "references-count": 30,
            },
        ],
    },
}

CROSSREF_DOI_RESPONSE = {
    "status": "ok",
    "message-type": "work",
    "message": CROSSREF_RESPONSE["message"]["items"][0],
}


class TestCrossrefProvider:
    """Tests for Crossref provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = CrossrefProvider({"type": "crossref"})

        assert provider.endpoint == "https://api.crossref.org"
        assert provider.min_interval_seconds == 0.5
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.email is None
        assert provider.api_key is None

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = CrossrefProvider({
            "type": "crossref",
            "email": "test@example.com",
            "api_key": "my-api-key",
            "timeout": 30,
            "min_interval_seconds": 1.0,
            "year_from": 2020,
            "year_to": 2024,
            "type": "journal-article",
            "has_full_text": True,
        })

        assert provider.email == "test@example.com"
        assert provider.api_key == "my-api-key"
        assert provider.timeout == 30
        assert provider.min_interval_seconds == 1.0
        assert provider.year_from == 2020
        assert provider.year_to == 2024
        assert provider.work_type == "journal-article"
        assert provider.has_full_text is True

    def test_is_available_always_true(self):
        """Test provider is always available (no API key required)."""
        provider = CrossrefProvider({})
        assert provider.is_available() is True

    def test_validate_config_valid_years(self):
        """Test validation passes for valid year range."""
        provider = CrossrefProvider({"year_from": 2020, "year_to": 2024})
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_year_from(self):
        """Test validation fails for invalid year_from."""
        provider = CrossrefProvider({"year_from": "invalid"})
        with pytest.raises(ValueError, match="Invalid year_from"):
            provider.validate_config()

    def test_validate_config_invalid_year_to(self):
        """Test validation fails for invalid year_to."""
        provider = CrossrefProvider({"year_to": "invalid"})
        with pytest.raises(ValueError, match="Invalid year_to"):
            provider.validate_config()

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = CrossrefProvider({})
        assert provider.get_name() == "Crossref"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = CrossrefProvider({})
        assert provider.name == "crossref"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = CROSSREF_RESPONSE
        mock_get.return_value = response

        provider = CrossrefProvider({"type": "crossref"})
        results = provider.search("deep learning NLP", max_results=2)

        assert len(results) == 2

        # Check first result
        assert results[0].title == "Deep Learning Methods for Natural Language Processing"
        assert results[0].url == "https://doi.org/10.1038/s41586-019-1099-1"
        assert "deep learning techniques" in results[0].snippet.lower()
        assert results[0].metadata["doi"] == "10.1038/s41586-019-1099-1"
        assert results[0].metadata["authors"] == ["John Smith", "Jane Doe"]
        assert results[0].metadata["year"] == 2023
        assert results[0].metadata["cited_by_count"] == 500
        assert results[0].metadata["venue"] == "Nature"
        assert results[0].metadata["publisher"] == "Nature Publishing Group"
        assert results[0].metadata["type"] == "journal-article"
        assert "citation" in results[0].metadata
        assert results[0].metadata["pdf_url"] == "https://www.nature.com/articles/s41586-019-1099-1.pdf"
        assert results[0].metadata["orcids"] == ["https://orcid.org/0000-0001-2345-6789"]

        # Check second result
        assert results[1].title == "Transformer Models: A Survey"
        assert results[1].metadata["year"] == 2022

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = CrossrefProvider({})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = CrossrefProvider({})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_includes_email_in_params(self, mock_get):
        """Test email is included in request params for polite pool."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({"email": "test@example.com"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["mailto"] == "test@example.com"

    @patch("requests.get")
    def test_search_includes_api_key_header(self, mock_get):
        """Test API key is included in request headers."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({"api_key": "my-api-key"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["headers"]["Crossref-Plus-API-Token"] == "Bearer my-api-key"

    @patch("requests.get")
    def test_search_with_author_prefix(self, mock_get):
        """Test search with author: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({})
        provider.search("author:John Smith", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["query.author"] == "John Smith"

    @patch("requests.get")
    def test_search_with_title_prefix(self, mock_get):
        """Test search with title: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({})
        provider.search("title:Deep Learning", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["query.title"] == "Deep Learning"

    @patch("requests.get")
    def test_search_with_doi_prefix(self, mock_get):
        """Test search with doi: prefix does DOI lookup."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = CROSSREF_DOI_RESPONSE
        mock_get.return_value = response

        provider = CrossrefProvider({})
        results = provider.search("doi:10.1038/s41586-019-1099-1", max_results=1)

        # Should call the works/{doi} endpoint
        call_args = mock_get.call_args
        assert "10.1038/s41586-019-1099-1" in call_args[0][0]
        assert len(results) == 1

    @patch("requests.get")
    def test_search_with_orcid_prefix(self, mock_get):
        """Test search with orcid: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({})
        provider.search("orcid:0000-0001-2345-6789", max_results=1)

        call_args = mock_get.call_args
        assert "orcid:0000-0001-2345-6789" in call_args[1]["params"]["filter"]

    @patch("requests.get")
    def test_search_with_year_filters(self, mock_get):
        """Test search includes year filters."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({
            "year_from": 2020,
            "year_to": 2024,
        })
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        filter_param = call_args[1]["params"]["filter"]
        assert "from-pub-date:2020" in filter_param
        assert "until-pub-date:2024" in filter_param

    @patch("requests.get")
    def test_search_with_type_filter(self, mock_get):
        """Test search includes type filter."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({"type": "journal-article"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert "type:journal-article" in call_args[1]["params"]["filter"]

    @patch("requests.get")
    def test_search_with_boolean_filters(self, mock_get):
        """Test search includes boolean filters."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"message": {"items": []}}
        mock_get.return_value = response

        provider = CrossrefProvider({
            "has_full_text": True,
            "has_references": True,
            "has_orcid": True,
        })
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        filter_param = call_args[1]["params"]["filter"]
        assert "has-full-text:true" in filter_param
        assert "has-references:true" in filter_param
        assert "has-orcid:true" in filter_param

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = CrossrefProvider({})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = CrossrefProvider({})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = CrossrefProvider({})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = CrossrefProvider({})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = CrossrefProvider({"snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = CrossrefProvider({"snippet_max_chars": 20})
        text = "This is a very long text that should be truncated"
        result = provider._truncate(text)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    @patch("requests.get")
    def test_score_calculation(self, mock_get):
        """Test score calculation includes citation boost."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/high",
                        "title": ["Highly Cited Paper"],
                        "is-referenced-by-count": 15000,
                        "author": [],
                    },
                    {
                        "DOI": "10.1234/low",
                        "title": ["Low Cited Paper"],
                        "is-referenced-by-count": 10,
                        "author": [],
                    }
                ]
            }
        }
        mock_get.return_value = response

        provider = CrossrefProvider({})
        results = provider.search("test", max_results=2)

        # Score is capped at 1.0 for highly cited first result
        assert results[0].score == 1.0
        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0

    @patch("requests.get")
    def test_get_work_by_doi(self, mock_get):
        """Test DOI lookup returns work details."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = CROSSREF_DOI_RESPONSE
        mock_get.return_value = response

        provider = CrossrefProvider({})
        work = provider.get_work_by_doi("10.1038/s41586-019-1099-1")

        assert work is not None
        assert work["DOI"] == "10.1038/s41586-019-1099-1"

    @patch("requests.get")
    def test_get_work_by_doi_with_url_prefix(self, mock_get):
        """Test DOI lookup handles doi.org URL prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = CROSSREF_DOI_RESPONSE
        mock_get.return_value = response

        provider = CrossrefProvider({})
        work = provider.get_work_by_doi("https://doi.org/10.1038/s41586-019-1099-1")

        assert work is not None
        # URL should have the DOI portion extracted
        call_args = mock_get.call_args
        assert "10.1038/s41586-019-1099-1" in call_args[0][0]

    @patch("requests.get")
    def test_get_work_by_doi_not_found(self, mock_get):
        """Test DOI lookup returns None for not found."""
        response = Mock()
        response.status_code = 404
        response.raise_for_status.side_effect = Exception("Not found")
        mock_get.return_value = response

        provider = CrossrefProvider({})
        work = provider.get_work_by_doi("10.1234/nonexistent")

        assert work is None

    def test_get_valid_types(self):
        """Test valid types include expected work types."""
        provider = CrossrefProvider({})
        valid_types = provider._get_valid_types()
        assert "journal-article" in valid_types
        assert "book" in valid_types
        assert "proceedings-article" in valid_types
        assert "dissertation" in valid_types
        assert "dataset" in valid_types

    @patch("requests.get")
    def test_abstract_html_stripping(self, mock_get):
        """Test that HTML is stripped from abstracts."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/test",
                        "title": ["Test Paper"],
                        "abstract": "<p>This is a <strong>test</strong> abstract.</p>",
                        "author": [],
                    }
                ]
            }
        }
        mock_get.return_value = response

        provider = CrossrefProvider({})
        results = provider.search("test", max_results=1)

        # HTML should be stripped
        assert "<p>" not in results[0].snippet
        assert "<strong>" not in results[0].snippet
        assert "This is a test abstract." in results[0].snippet
