"""
Tests for Semantic Scholar provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.remote.providers.academic.semantic_scholar import SemanticScholarProvider


SEMANTIC_SCHOLAR_RESPONSE = {
    "total": 2,
    "offset": 0,
    "data": [
        {
            "paperId": "abc123",
            "title": "Deep Learning for Natural Language Processing",
            "abstract": "This paper presents a comprehensive survey of deep learning techniques for NLP tasks.",
            "url": "https://www.semanticscholar.org/paper/abc123",
            "year": 2023,
            "authors": [
                {"authorId": "1", "name": "Jane Smith"},
                {"authorId": "2", "name": "John Doe"},
            ],
            "citationCount": 150,
            "influentialCitationCount": 25,
            "venue": "ACL",
            "publicationDate": "2023-05-15",
            "externalIds": {"DOI": "10.1234/example.doi", "ArXiv": "2301.12345"},
            "isOpenAccess": True,
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2301.12345.pdf"},
            "fieldsOfStudy": ["Computer Science"],
        },
        {
            "paperId": "def456",
            "title": "Transformers in Machine Learning",
            "abstract": "An introduction to transformer architectures.",
            "url": "https://www.semanticscholar.org/paper/def456",
            "year": 2022,
            "authors": [{"authorId": "3", "name": "Alice Johnson"}],
            "citationCount": 50,
            "influentialCitationCount": 5,
            "venue": "NeurIPS",
            "publicationDate": "2022-12-01",
            "externalIds": {"DOI": "10.5678/another.doi"},
            "isOpenAccess": False,
            "openAccessPdf": None,
            "fieldsOfStudy": ["Computer Science", "Mathematics"],
        },
    ],
}


class TestSemanticScholarProvider:
    """Tests for Semantic Scholar provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = SemanticScholarProvider({"type": "semantic_scholar"})

        assert provider.endpoint == "https://api.semanticscholar.org/graph/v1"
        assert provider.min_interval_seconds == 1.0
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.api_key is None
        assert provider.fields_of_study == []

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = SemanticScholarProvider({
            "type": "semantic_scholar",
            "api_key": "test-key",
            "timeout": 30,
            "min_interval_seconds": 2.0,
            "fields_of_study": ["Computer Science"],
            "year_range": "2020-2024",
            "open_access_only": True,
        })

        assert provider.api_key == "test-key"
        assert provider.timeout == 30
        assert provider.min_interval_seconds == 2.0
        assert provider.fields_of_study == ["Computer Science"]
        assert provider.year_range == "2020-2024"
        assert provider.open_access_only is True

    def test_is_available_always_true(self):
        """Test provider is always available (API works without key)."""
        provider = SemanticScholarProvider({})
        assert provider.is_available() is True

    def test_validate_config_valid_year_range(self):
        """Test validation passes for valid year range."""
        provider = SemanticScholarProvider({"year_range": "2020-2024"})
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_year_range(self):
        """Test validation fails for invalid year range."""
        provider = SemanticScholarProvider({"year_range": "invalid"})
        with pytest.raises(ValueError, match="Invalid year_range format"):
            provider.validate_config()

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = SemanticScholarProvider({})
        assert provider.get_name() == "Semantic Scholar"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = SemanticScholarProvider({})
        assert provider.name == "semantic_scholar"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = SEMANTIC_SCHOLAR_RESPONSE
        mock_get.return_value = response

        provider = SemanticScholarProvider({"type": "semantic_scholar"})
        results = provider.search("deep learning", max_results=2)

        assert len(results) == 2

        # Check first result
        assert results[0].title == "Deep Learning for Natural Language Processing"
        assert results[0].url == "https://www.semanticscholar.org/paper/abc123"
        assert "survey of deep learning" in results[0].snippet
        assert results[0].metadata["paper_id"] == "abc123"
        assert results[0].metadata["authors"] == ["Jane Smith", "John Doe"]
        assert results[0].metadata["year"] == 2023
        assert results[0].metadata["citation_count"] == 150
        assert results[0].metadata["influential_citation_count"] == 25
        assert results[0].metadata["doi"] == "10.1234/example.doi"
        assert results[0].metadata["arxiv_id"] == "2301.12345"
        assert results[0].metadata["is_open_access"] is True
        assert results[0].metadata["pdf_url"] == "https://arxiv.org/pdf/2301.12345.pdf"
        assert "citation" in results[0].metadata

        # Check second result
        assert results[1].title == "Transformers in Machine Learning"
        assert results[1].metadata["is_open_access"] is False
        assert results[1].metadata["pdf_url"] is None

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = SemanticScholarProvider({})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = SemanticScholarProvider({})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_includes_api_key_header(self, mock_get):
        """Test API key is included in request headers."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": []}
        mock_get.return_value = response

        provider = SemanticScholarProvider({"api_key": "my-api-key"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["headers"]["x-api-key"] == "my-api-key"

    @patch("requests.get")
    def test_search_with_fields_of_study_filter(self, mock_get):
        """Test search includes fields of study filter."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": []}
        mock_get.return_value = response

        provider = SemanticScholarProvider({
            "fields_of_study": ["Computer Science", "Mathematics"]
        })
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert "fieldsOfStudy" in call_args[1]["params"]
        assert call_args[1]["params"]["fieldsOfStudy"] == "Computer Science,Mathematics"

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = SemanticScholarProvider({})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = SemanticScholarProvider({})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = SemanticScholarProvider({})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = SemanticScholarProvider({})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = SemanticScholarProvider({"snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = SemanticScholarProvider({"snippet_max_chars": 20})
        text = "This is a very long text that should be truncated"
        result = provider._truncate(text)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    def test_parse_year_range_valid(self):
        """Test parsing valid year range."""
        provider = SemanticScholarProvider({})
        assert provider._parse_year_range("2020-2024") == (2020, 2024)
        assert provider._parse_year_range("2020-") == (2020, None)

    def test_parse_year_range_invalid(self):
        """Test parsing invalid year range."""
        provider = SemanticScholarProvider({})
        assert provider._parse_year_range("invalid") is None
        assert provider._parse_year_range("2020") is None

    @patch("requests.get")
    def test_score_calculation(self, mock_get):
        """Test score calculation includes citation boost."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "data": [
                {
                    "paperId": "high-cited",
                    "title": "Highly Cited Paper",
                    "abstract": "Abstract",
                    "citationCount": 10000,
                    "influentialCitationCount": 100,
                    "authors": [],
                },
                {
                    "paperId": "low-cited",
                    "title": "Low Cited Paper",
                    "abstract": "Abstract",
                    "citationCount": 10,
                    "influentialCitationCount": 1,
                    "authors": [],
                }
            ]
        }
        mock_get.return_value = response

        provider = SemanticScholarProvider({})
        results = provider.search("test", max_results=2)

        # Score is capped at 1.0, but highly cited paper should have higher score
        # First position + high citations = max 1.0
        assert results[0].score == 1.0  # Capped at 1.0
        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0
