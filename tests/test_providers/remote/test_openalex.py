"""
Tests for OpenAlex provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.query.remote.openalex import OpenAlexProvider


OPENALEX_RESPONSE = {
    "meta": {
        "count": 2,
        "db_response_time_ms": 50,
        "page": 1,
        "per_page": 10,
    },
    "results": [
        {
            "id": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.1038/s41586-019-1099-1",
            "title": "Attention is All You Need: Transformers in Deep Learning",
            "abstract_inverted_index": {
                "This": [0],
                "paper": [1],
                "introduces": [2],
                "the": [3, 10],
                "transformer": [4],
                "architecture": [5],
                "for": [6],
                "sequence": [7],
                "to": [8],
                "sequence": [9],
                "modeling.": [11],
            },
            "publication_year": 2023,
            "publication_date": "2023-06-15",
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A5023888391",
                        "display_name": "John Smith",
                    },
                    "institutions": [],
                },
                {
                    "author": {
                        "id": "https://openalex.org/A5023888392",
                        "display_name": "Jane Doe",
                    },
                    "institutions": [],
                },
            ],
            "cited_by_count": 5000,
            "type": "article",
            "open_access": {
                "is_oa": True,
                "oa_url": "https://arxiv.org/pdf/1706.03762.pdf",
            },
            "primary_location": {
                "source": {
                    "display_name": "Nature",
                },
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
            },
            "topics": [
                {"display_name": "Deep Learning"},
                {"display_name": "Natural Language Processing"},
            ],
            "concepts": [
                {"display_name": "Artificial Intelligence"},
                {"display_name": "Machine Learning"},
            ],
        },
        {
            "id": "https://openalex.org/W2741809808",
            "doi": "https://doi.org/10.1234/example.2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract_inverted_index": {
                "BERT": [0],
                "is": [1],
                "a": [2],
                "language": [3],
                "model.": [4],
            },
            "publication_year": 2022,
            "publication_date": "2022-10-01",
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A5023888393",
                        "display_name": "Alice Johnson",
                    },
                    "institutions": [],
                },
            ],
            "cited_by_count": 1000,
            "type": "article",
            "open_access": {
                "is_oa": False,
                "oa_url": None,
            },
            "primary_location": {
                "source": {
                    "display_name": "ACL",
                },
                "pdf_url": None,
            },
            "topics": [
                {"display_name": "Natural Language Processing"},
            ],
            "concepts": [
                {"display_name": "Machine Learning"},
            ],
        },
    ],
}


class TestOpenAlexProvider:
    """Tests for OpenAlex provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = OpenAlexProvider({"type": "openalex"})

        assert provider.endpoint == "https://api.openalex.org"
        assert provider.min_interval_seconds == 0.1
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.email is None
        assert provider.concepts == []

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = OpenAlexProvider({
            "type": "openalex",
            "email": "test@example.com",
            "timeout": 30,
            "min_interval_seconds": 0.5,
            "year_from": 2020,
            "year_to": 2024,
            "open_access_only": True,
        })

        assert provider.email == "test@example.com"
        assert provider.timeout == 30
        assert provider.min_interval_seconds == 0.5
        assert provider.year_from == 2020
        assert provider.year_to == 2024
        assert provider.open_access_only is True

    def test_is_available_always_true(self):
        """Test provider is always available (no API key required)."""
        provider = OpenAlexProvider({})
        assert provider.is_available() is True

    def test_validate_config_valid_years(self):
        """Test validation passes for valid year range."""
        provider = OpenAlexProvider({"year_from": 2020, "year_to": 2024})
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_year_from(self):
        """Test validation fails for invalid year_from."""
        provider = OpenAlexProvider({"year_from": "invalid"})
        with pytest.raises(ValueError, match="Invalid year_from"):
            provider.validate_config()

    def test_validate_config_invalid_year_to(self):
        """Test validation fails for invalid year_to."""
        provider = OpenAlexProvider({"year_to": "invalid"})
        with pytest.raises(ValueError, match="Invalid year_to"):
            provider.validate_config()

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = OpenAlexProvider({})
        assert provider.get_name() == "OpenAlex"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = OpenAlexProvider({})
        assert provider.name == "openalex"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = OPENALEX_RESPONSE
        mock_get.return_value = response

        provider = OpenAlexProvider({"type": "openalex"})
        results = provider.search("transformers deep learning", max_results=2)

        assert len(results) == 2

        # Check first result
        assert results[0].title == "Attention is All You Need: Transformers in Deep Learning"
        assert results[0].url == "https://doi.org/10.1038/s41586-019-1099-1"
        assert "transformer architecture" in results[0].snippet
        assert results[0].metadata["openalex_id"] == "https://openalex.org/W2741809807"
        assert results[0].metadata["authors"] == ["John Smith", "Jane Doe"]
        assert results[0].metadata["year"] == 2023
        assert results[0].metadata["cited_by_count"] == 5000
        assert results[0].metadata["is_open_access"] is True
        assert results[0].metadata["doi"] == "https://doi.org/10.1038/s41586-019-1099-1"
        assert "citation" in results[0].metadata

        # Check second result
        assert results[1].title == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert results[1].metadata["is_open_access"] is False

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = OpenAlexProvider({})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = OpenAlexProvider({})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_includes_email_in_params(self, mock_get):
        """Test email is included in request params for polite pool."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({"email": "test@example.com"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["mailto"] == "test@example.com"

    @patch("requests.get")
    def test_search_with_author_prefix(self, mock_get):
        """Test search with author: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({})
        provider.search("author:John Smith", max_results=1)

        call_args = mock_get.call_args
        assert "authorships.author.display_name.search:John Smith" in call_args[1]["params"]["filter"]

    @patch("requests.get")
    def test_search_with_title_prefix(self, mock_get):
        """Test search with title: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({})
        provider.search("title:Deep Learning", max_results=1)

        call_args = mock_get.call_args
        assert "title.search:Deep Learning" in call_args[1]["params"]["filter"]

    @patch("requests.get")
    def test_search_with_doi_prefix(self, mock_get):
        """Test search with doi: prefix."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({})
        provider.search("doi:10.1234/example", max_results=1)

        call_args = mock_get.call_args
        assert "doi:https://doi.org/10.1234/example" in call_args[1]["params"]["filter"]

    @patch("requests.get")
    def test_search_with_year_filters(self, mock_get):
        """Test search includes year filters."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({
            "year_from": 2020,
            "year_to": 2024,
        })
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        filter_param = call_args[1]["params"]["filter"]
        assert "publication_year:>2019" in filter_param
        assert "publication_year:<2025" in filter_param

    @patch("requests.get")
    def test_search_with_open_access_filter(self, mock_get):
        """Test search includes open access filter."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = OpenAlexProvider({"open_access_only": True})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert "open_access.is_oa:true" in call_args[1]["params"]["filter"]

    def test_reconstruct_abstract(self):
        """Test abstract reconstruction from inverted index."""
        provider = OpenAlexProvider({})
        inverted_index = {
            "Hello": [0],
            "world": [1],
            "this": [2],
            "is": [3],
            "a": [4],
            "test": [5],
        }
        result = provider._reconstruct_abstract(inverted_index)
        assert result == "Hello world this is a test"

    def test_reconstruct_abstract_empty(self):
        """Test abstract reconstruction with empty input."""
        provider = OpenAlexProvider({})
        assert provider._reconstruct_abstract(None) == ""
        assert provider._reconstruct_abstract({}) == ""

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = OpenAlexProvider({})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = OpenAlexProvider({})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = OpenAlexProvider({})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = OpenAlexProvider({})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = OpenAlexProvider({"snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = OpenAlexProvider({"snippet_max_chars": 20})
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
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Highly Cited Paper",
                    "abstract_inverted_index": {"Abstract": [0]},
                    "cited_by_count": 15000,
                    "authorships": [],
                    "open_access": {"is_oa": False},
                },
                {
                    "id": "https://openalex.org/W2",
                    "title": "Low Cited Paper",
                    "abstract_inverted_index": {"Abstract": [0]},
                    "cited_by_count": 10,
                    "authorships": [],
                    "open_access": {"is_oa": False},
                }
            ]
        }
        mock_get.return_value = response

        provider = OpenAlexProvider({})
        results = provider.search("test", max_results=2)

        # Score is capped at 1.0 for highly cited first result
        assert results[0].score == 1.0
        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0

    def test_get_valid_types(self):
        """Test valid types include expected work types."""
        provider = OpenAlexProvider({})
        valid_types = provider._get_valid_types()
        assert "article" in valid_types
        assert "book" in valid_types
        assert "dataset" in valid_types
        assert "dissertation" in valid_types
