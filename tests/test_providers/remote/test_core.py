"""
Tests for CORE provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.remote.providers.academic.core import COREProvider


CORE_SEARCH_RESPONSE = {
    "totalHits": 2,
    "results": [
        {
            "id": "12345",
            "title": "Open Access in Academic Research",
            "abstract": "This paper discusses the importance of open access publishing in academic research.",
            "yearPublished": 2023,
            "authors": [
                {"name": "Jane Smith"},
                {"name": "John Doe"},
            ],
            "downloadUrl": "https://core.ac.uk/download/pdf/12345.pdf",
            "identifiers": [
                {"type": "doi", "identifier": "10.1234/oa.example"},
            ],
            "repositories": [
                {"id": "1", "name": "MIT Repository"},
            ],
            "fullText": "Full text content here...",
            "fullTextUrl": "https://core.ac.uk/reader/12345",
            "language": {"code": "en", "name": "English"},
            "publisher": "Open Access Publishers",
            "journals": [{"title": "Journal of Open Science"}],
        },
        {
            "id": "67890",
            "title": "Repository Aggregation Methods",
            "abstract": "A study of methods for aggregating academic repositories.",
            "yearPublished": 2022,
            "authors": [{"name": "Alice Johnson"}],
            "downloadUrl": None,
            "identifiers": [],
            "repositories": [
                {"id": "2", "name": "Harvard Repository"},
                {"id": "3", "name": "Stanford Repository"},
            ],
            "fullText": None,
            "fullTextUrl": None,
            "language": {"code": "en", "name": "English"},
            "publisher": None,
            "journals": [],
        },
    ],
}


class TestCOREProvider:
    """Tests for CORE provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = COREProvider({"type": "core", "api_key": "test-key"})

        assert provider.endpoint == "https://api.core.ac.uk/v3"
        assert provider.min_interval_seconds == 1.0
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.repository_ids == []
        assert provider.year_from is None
        assert provider.year_to is None
        assert provider.full_text_only is False
        assert provider.language is None

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = COREProvider({
            "type": "core",
            "api_key": "test-key",
            "timeout": 30,
            "min_interval_seconds": 2.0,
            "repository_ids": ["1", "2"],
            "year_from": 2020,
            "year_to": 2024,
            "full_text_only": True,
            "language": "en",
        })

        assert provider.api_key == "test-key"
        assert provider.timeout == 30
        assert provider.min_interval_seconds == 2.0
        assert provider.repository_ids == ["1", "2"]
        assert provider.year_from == 2020
        assert provider.year_to == 2024
        assert provider.full_text_only is True
        assert provider.language == "en"

    def test_is_available_with_api_key(self):
        """Test provider is available when API key is set."""
        provider = COREProvider({"api_key": "test-key"})
        assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test provider is not available without API key."""
        provider = COREProvider({})
        assert provider.is_available() is False

    def test_validate_config_requires_api_key(self):
        """Test validation fails without API key."""
        provider = COREProvider({})
        with pytest.raises(ValueError, match="CORE provider requires an API key"):
            provider.validate_config()

    def test_validate_config_invalid_year_range(self):
        """Test validation fails when year_from > year_to."""
        provider = COREProvider({
            "api_key": "test-key",
            "year_from": 2024,
            "year_to": 2020,
        })
        with pytest.raises(ValueError, match="year_from cannot be greater than year_to"):
            provider.validate_config()

    def test_validate_config_valid(self):
        """Test validation passes with valid config."""
        provider = COREProvider({
            "api_key": "test-key",
            "year_from": 2020,
            "year_to": 2024,
        })
        provider.validate_config()  # Should not raise

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = COREProvider({"api_key": "test-key"})
        assert provider.get_name() == "CORE"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = COREProvider({"api_key": "test-key"})
        assert provider.name == "core"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = CORE_SEARCH_RESPONSE
        mock_get.return_value = response

        provider = COREProvider({"api_key": "test-key"})
        results = provider.search("open access", max_results=2)

        assert len(results) == 2

        # Check first result
        assert results[0].title == "Open Access in Academic Research"
        assert results[0].url == "https://core.ac.uk/download/pdf/12345.pdf"
        assert "importance of open access" in results[0].snippet
        assert results[0].metadata["core_id"] == "12345"
        assert results[0].metadata["authors"] == ["Jane Smith", "John Doe"]
        assert results[0].metadata["year"] == 2023
        assert results[0].metadata["doi"] == "10.1234/oa.example"
        assert results[0].metadata["source_id"] == "10.1234/oa.example"
        assert results[0].metadata["repositories"] == ["MIT Repository"]
        assert results[0].metadata["has_full_text"] is True
        assert results[0].metadata["language"] == "en"
        assert results[0].metadata["publisher"] == "Open Access Publishers"
        assert results[0].metadata["journal"] == "Journal of Open Science"
        assert "citation" in results[0].metadata

        # Check second result (no download URL, uses CORE page URL)
        assert results[1].title == "Repository Aggregation Methods"
        assert results[1].url == "https://core.ac.uk/works/67890"
        assert results[1].metadata["has_full_text"] is False
        assert results[1].metadata["doi"] is None

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = COREProvider({"api_key": "test-key"})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = COREProvider({"api_key": "test-key"})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_includes_auth_header(self, mock_get):
        """Test API key is included in Authorization header."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"results": []}
        mock_get.return_value = response

        provider = COREProvider({"api_key": "my-api-key"})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer my-api-key"

    def test_build_query_basic(self):
        """Test basic query building."""
        provider = COREProvider({"api_key": "test-key"})
        query = provider._build_query("machine learning")
        assert query == "(machine learning)"

    def test_build_query_with_year_filter(self):
        """Test query building with year filter."""
        provider = COREProvider({
            "api_key": "test-key",
            "year_from": 2020,
            "year_to": 2024,
        })
        query = provider._build_query("test")
        assert "(yearPublished>=2020)" in query
        assert "(yearPublished<=2024)" in query

    def test_build_query_with_full_text_filter(self):
        """Test query building with full text filter."""
        provider = COREProvider({
            "api_key": "test-key",
            "full_text_only": True,
        })
        query = provider._build_query("test")
        assert "(fullText:*)" in query

    def test_build_query_with_language_filter(self):
        """Test query building with language filter."""
        provider = COREProvider({
            "api_key": "test-key",
            "language": "en",
        })
        query = provider._build_query("test")
        assert "(language.code:en)" in query

    def test_build_query_with_repository_filter(self):
        """Test query building with repository filter."""
        provider = COREProvider({
            "api_key": "test-key",
            "repository_ids": ["1", "2"],
        })
        query = provider._build_query("test")
        assert "repositories.id:1" in query
        assert "repositories.id:2" in query

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = COREProvider({"api_key": "test-key"})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = COREProvider({"api_key": "test-key"})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = COREProvider({"api_key": "test-key"})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = COREProvider({"api_key": "test-key"})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = COREProvider({"api_key": "test-key", "snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = COREProvider({"api_key": "test-key", "snippet_max_chars": 20})
        text = "This is a very long text that should be truncated"
        result = provider._truncate(text)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    @patch("requests.get")
    def test_get_work_details(self, mock_get):
        """Test fetching work details."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"id": "12345", "title": "Test Work"}
        mock_get.return_value = response

        provider = COREProvider({"api_key": "test-key"})
        details = provider.get_work_details("12345")

        assert details is not None
        assert details["id"] == "12345"
        assert details["title"] == "Test Work"

    @patch("requests.get")
    def test_get_work_details_error(self, mock_get):
        """Test work details handles errors."""
        mock_get.side_effect = Exception("API Error")

        provider = COREProvider({"api_key": "test-key"})
        details = provider.get_work_details("12345")

        assert details is None

    @patch("requests.get")
    def test_score_calculation(self, mock_get):
        """Test score calculation based on position."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "results": [
                {"id": "1", "title": "First", "authors": []},
                {"id": "2", "title": "Second", "authors": []},
                {"id": "3", "title": "Third", "authors": []},
            ]
        }
        mock_get.return_value = response

        provider = COREProvider({"api_key": "test-key"})
        results = provider.search("test", max_results=3)

        # First result should have highest score
        assert results[0].score > results[1].score
        assert results[1].score > results[2].score
        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0
