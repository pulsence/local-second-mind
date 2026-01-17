"""
Tests for IxTheo provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.query.remote.ixtheo import IxTheoProvider


# Sample RSS response from IxTheo
IXTHEO_RSS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <channel>
        <title>IxTheo Search Results</title>
        <link>https://ixtheo.de</link>
        <description>Index Theologicus search results</description>
        <item>
            <title>Theological Hermeneutics and Biblical Interpretation</title>
            <link>https://ixtheo.de/Record/1234567890</link>
            <description>This article examines the relationship between theological hermeneutics and contemporary biblical interpretation methods.</description>
            <dc:creator>Thomas Müller</dc:creator>
            <dc:creator>Anna Schmidt</dc:creator>
            <pubDate>2023-05-15</pubDate>
            <category>Biblical Studies</category>
            <category>Hermeneutics</category>
        </item>
        <item>
            <title>The Concept of Grace in Reformation Theology</title>
            <link>https://ixtheo.de/Record/0987654321</link>
            <description>A comprehensive study of how the concept of grace was understood and debated during the Protestant Reformation.</description>
            <dc:creator>Martin Weber</dc:creator>
            <pubDate>2022-11-20</pubDate>
            <category>Reformation</category>
            <category>Protestant Theology</category>
            <category>Grace</category>
        </item>
        <item>
            <title>Interreligious Dialogue in the 21st Century</title>
            <link>https://ixtheo.de/Record/1122334455</link>
            <description>An analysis of contemporary approaches to interreligious dialogue between Christianity, Judaism, and Islam.</description>
            <author>Sarah Cohen</author>
            <pubDate>2024-01-10</pubDate>
            <category>Interreligious Dialogue</category>
            <category>Comparative Religion</category>
        </item>
    </channel>
</rss>"""


IXTHEO_EMPTY_RSS_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>IxTheo Search Results</title>
        <link>https://ixtheo.de</link>
        <description>No results found</description>
    </channel>
</rss>"""


class TestIxTheoProvider:
    """Tests for IxTheo provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = IxTheoProvider({"type": "ixtheo"})

        assert provider.endpoint == "https://ixtheo.de/Search/Results"
        assert provider.min_interval_seconds == 1.0
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.language == "en"
        assert provider.traditions == []
        assert provider.search_type == "all"
        assert provider.include_reviews is True

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = IxTheoProvider({
            "type": "ixtheo",
            "timeout": 30,
            "min_interval_seconds": 2.0,
            "language": "de",
            "traditions": ["catholic", "protestant"],
            "search_type": "title",
            "include_reviews": False,
            "year_from": 2020,
            "year_to": 2024,
        })

        assert provider.timeout == 30
        assert provider.min_interval_seconds == 2.0
        assert provider.language == "de"
        assert provider.traditions == ["catholic", "protestant"]
        assert provider.search_type == "title"
        assert provider.include_reviews is False
        assert provider.year_from == 2020
        assert provider.year_to == 2024

    def test_is_available_always_true(self):
        """Test provider is always available (no API key needed)."""
        provider = IxTheoProvider({})
        assert provider.is_available() is True

    def test_validate_config_valid_search_type(self):
        """Test validation passes for valid search type."""
        provider = IxTheoProvider({"search_type": "author"})
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_search_type(self):
        """Test validation fails for invalid search type."""
        provider = IxTheoProvider({"search_type": "invalid_type"})
        with pytest.raises(ValueError, match="Invalid search_type"):
            provider.validate_config()

    def test_validate_config_valid_language(self):
        """Test validation passes for valid language."""
        provider = IxTheoProvider({"language": "de"})
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_language(self):
        """Test validation fails for invalid language."""
        provider = IxTheoProvider({"language": "invalid"})
        with pytest.raises(ValueError, match="Invalid language"):
            provider.validate_config()

    def test_validate_config_valid_traditions(self):
        """Test validation passes for valid traditions."""
        provider = IxTheoProvider({
            "traditions": ["catholic", "protestant", "jewish"]
        })
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_tradition(self):
        """Test validation fails for invalid tradition."""
        provider = IxTheoProvider({"traditions": ["invalid_tradition"]})
        with pytest.raises(ValueError, match="Invalid traditions"):
            provider.validate_config()

    def test_validate_config_year_range_error(self):
        """Test validation fails when year_from > year_to."""
        provider = IxTheoProvider({
            "year_from": 2024,
            "year_to": 2020,
        })
        with pytest.raises(ValueError, match="year_from cannot be greater"):
            provider.validate_config()

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = IxTheoProvider({})
        assert provider.get_name() == "Index Theologicus"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = IxTheoProvider({})
        assert provider.name == "ixtheo"

    def test_list_traditions(self):
        """Test listing traditions."""
        provider = IxTheoProvider({})
        traditions = provider.list_traditions()

        assert "catholic" in traditions
        assert "protestant" in traditions
        assert "jewish" in traditions
        assert "islamic" in traditions
        assert traditions["catholic"] == "Catholic"

    def test_list_languages(self):
        """Test listing languages."""
        provider = IxTheoProvider({})
        languages = provider.list_languages()

        assert "en" in languages
        assert "de" in languages
        assert "la" in languages
        assert "he" in languages
        assert languages["en"] == "English"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({"type": "ixtheo"})
        results = provider.search("hermeneutics", max_results=5)

        assert len(results) == 3

        # Check first result
        assert results[0].title == "Theological Hermeneutics and Biblical Interpretation"
        assert "ixtheo.de" in results[0].url
        assert "theological hermeneutics" in results[0].snippet.lower()
        assert results[0].metadata["ixtheo_id"] == "1234567890"
        assert results[0].metadata["authors"] == ["Thomas Müller", "Anna Schmidt"]
        assert results[0].metadata["year"] == 2023
        assert "Biblical Studies" in results[0].metadata["subjects"]
        assert "citation" in results[0].metadata

        # Check second result
        assert results[1].title == "The Concept of Grace in Reformation Theology"
        assert results[1].metadata["authors"] == ["Martin Weber"]

        # Check third result (uses author tag instead of dc:creator)
        assert results[2].title == "Interreligious Dialogue in the 21st Century"
        assert results[2].metadata["authors"] == ["Sarah Cohen"]

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = IxTheoProvider({})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = IxTheoProvider({})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_handles_empty_results(self, mock_get):
        """Test search handles empty results."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_EMPTY_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search("nonexistent query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_with_author_prefix(self, mock_get):
        """Test search with author: prefix."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search("author:Müller", max_results=5)

        # Verify the search type is set correctly
        call_args = mock_get.call_args
        assert call_args[1]["params"]["type"] == "Author"

    @patch("requests.get")
    def test_search_with_title_prefix(self, mock_get):
        """Test search with title: prefix."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search("title:Reformation", max_results=5)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["type"] == "Title"

    @patch("requests.get")
    def test_search_with_subject_prefix(self, mock_get):
        """Test search with subject: prefix."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search("subject:hermeneutics", max_results=5)

        call_args = mock_get.call_args
        assert call_args[1]["params"]["type"] == "Subject"

    @patch("requests.get")
    def test_search_bible_passage(self, mock_get):
        """Test searching for Bible passage."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search_bible_passage("Mt 5:1-12", max_results=10)

        # Verify the method was called
        assert mock_get.called
        call_args = mock_get.call_args
        # Bible passage should be quoted
        assert '"Mt 5:1-12"' in call_args[1]["params"]["lookfor"]

    @patch("requests.get")
    def test_search_by_tradition(self, mock_get):
        """Test searching by tradition."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search_by_tradition("catholic", "grace", max_results=10)

        assert mock_get.called

    def test_search_by_tradition_invalid(self):
        """Test searching with invalid tradition."""
        provider = IxTheoProvider({})
        results = provider.search_by_tradition("invalid_tradition", max_results=5)

        assert results == []

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = IxTheoProvider({})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = IxTheoProvider({})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = IxTheoProvider({})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = IxTheoProvider({})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = IxTheoProvider({"snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = IxTheoProvider({"snippet_max_chars": 20})
        text = "This is a very long text that should be truncated"
        result = provider._truncate(text)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    def test_extract_year_valid(self):
        """Test year extraction from date strings."""
        provider = IxTheoProvider({})
        assert provider._extract_year("2023") == 2023
        assert provider._extract_year("2023-06-15") == 2023
        assert provider._extract_year("Published in 2021") == 2021

    def test_extract_year_invalid(self):
        """Test year extraction with invalid input."""
        provider = IxTheoProvider({})
        assert provider._extract_year("") is None
        assert provider._extract_year("no year here") is None

    def test_extract_record_id(self):
        """Test record ID extraction from URLs."""
        provider = IxTheoProvider({})
        assert provider._extract_record_id("https://ixtheo.de/Record/1234567890") == "1234567890"
        assert provider._extract_record_id("/Record/ABC123") == "ABC123"
        assert provider._extract_record_id("https://ixtheo.de/Record/XYZ?view=full") == "XYZ"
        assert provider._extract_record_id("no record here") == ""

    def test_clean_html(self):
        """Test HTML cleaning."""
        provider = IxTheoProvider({})
        assert provider._clean_html("<p>Hello <b>World</b></p>") == "Hello World"
        assert provider._clean_html("No HTML here") == "No HTML here"
        assert provider._clean_html("  Multiple   spaces  ") == "Multiple spaces"

    def test_parse_query_plain_text(self):
        """Test query parsing with plain text."""
        provider = IxTheoProvider({})
        parsed = provider._parse_query("theology hermeneutics")

        assert parsed["lookfor"] == "theology hermeneutics"
        assert parsed["type"] is None

    def test_parse_query_with_author(self):
        """Test query parsing with author prefix."""
        provider = IxTheoProvider({})
        parsed = provider._parse_query("author:Müller")

        assert parsed["lookfor"] == "Müller"
        assert parsed["type"] == "author"

    def test_parse_query_with_title(self):
        """Test query parsing with title prefix."""
        provider = IxTheoProvider({})
        parsed = provider._parse_query("title:Reformation Grace")

        assert parsed["lookfor"] == "Reformation Grace"
        assert parsed["type"] == "title"

    @patch("requests.get")
    def test_score_calculation(self, mock_get):
        """Test score calculation is position-based."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        results = provider.search("test", max_results=5)

        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0

        # First result should have highest score
        if len(results) > 1:
            assert results[0].score > results[-1].score

    @patch("requests.get")
    def test_request_includes_headers(self, mock_get):
        """Test request includes proper headers."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({})
        provider.search("test", max_results=1)

        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "User-Agent" in headers
        assert "LocalSecondMind" in headers["User-Agent"]

    @patch("requests.get")
    def test_build_search_params_with_filters(self, mock_get):
        """Test search params include configured filters."""
        response = Mock()
        response.status_code = 200
        response.text = IXTHEO_RSS_RESPONSE
        mock_get.return_value = response

        provider = IxTheoProvider({
            "language": "de",
            "year_from": 2020,
            "year_to": 2024,
            "traditions": ["catholic"],
        })
        provider.search("grace", max_results=5)

        # Verify filters are passed
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        assert "filter[]" in params

    def test_search_types_mapping(self):
        """Test search type mapping is correct."""
        provider = IxTheoProvider({})

        assert provider.SEARCH_TYPES["all"] == "AllFields"
        assert provider.SEARCH_TYPES["title"] == "Title"
        assert provider.SEARCH_TYPES["author"] == "Author"
        assert provider.SEARCH_TYPES["subject"] == "Subject"
