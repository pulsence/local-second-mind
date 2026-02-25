"""
Tests for PhilPapers provider implementation.
"""

from unittest.mock import Mock, patch
import pytest

from lsm.remote.providers.academic.philpapers import PhilPapersProvider


# Sample OAI-PMH ListRecords response
OAI_PMH_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.openarchives.org/OAI/2.0/ http://www.openarchives.org/OAI/2.0/OAI-PMH.xsd">
    <responseDate>2024-01-15T12:00:00Z</responseDate>
    <request verb="ListRecords" metadataPrefix="oai_dc">https://philpapers.org/oai.pl</request>
    <ListRecords>
        <record>
            <header>
                <identifier>oai:philpapers.org:SMIETH-1</identifier>
                <datestamp>2023-06-15</datestamp>
            </header>
            <metadata>
                <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                           xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:title>The Problem of Knowledge in Epistemology</dc:title>
                    <dc:creator>John Smith</dc:creator>
                    <dc:creator>Jane Doe</dc:creator>
                    <dc:subject>Epistemology</dc:subject>
                    <dc:subject>Knowledge</dc:subject>
                    <dc:description>This paper examines the fundamental nature of knowledge and justified belief in contemporary epistemology.</dc:description>
                    <dc:date>2023</dc:date>
                    <dc:type>Article</dc:type>
                    <dc:identifier>https://philpapers.org/rec/SMIETH-1</dc:identifier>
                    <dc:identifier>10.1234/example.doi</dc:identifier>
                    <dc:publisher>Journal of Philosophy</dc:publisher>
                    <dc:source>Journal of Philosophy, Vol. 120</dc:source>
                </oai_dc:dc>
            </metadata>
        </record>
        <record>
            <header>
                <identifier>oai:philpapers.org:JONESM-2</identifier>
                <datestamp>2022-12-01</datestamp>
            </header>
            <metadata>
                <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/"
                           xmlns:dc="http://purl.org/dc/elements/1.1/">
                    <dc:title>Virtue Ethics and Moral Character</dc:title>
                    <dc:creator>Alice Jones</dc:creator>
                    <dc:subject>Ethics</dc:subject>
                    <dc:subject>Virtue Ethics</dc:subject>
                    <dc:description>An exploration of virtue ethics and its implications for understanding moral character.</dc:description>
                    <dc:date>2022-10</dc:date>
                    <dc:type>Article</dc:type>
                    <dc:identifier>https://philpapers.org/rec/JONESM-2</dc:identifier>
                    <dc:publisher>Ethics Journal</dc:publisher>
                </oai_dc:dc>
            </metadata>
        </record>
    </ListRecords>
</OAI-PMH>"""


OAI_PMH_ERROR_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
    <responseDate>2024-01-15T12:00:00Z</responseDate>
    <request>https://philpapers.org/oai.pl</request>
    <error code="noRecordsMatch">No records match the query</error>
</OAI-PMH>"""


class TestPhilPapersProvider:
    """Tests for PhilPapers provider."""

    def test_provider_initialization_defaults(self):
        """Test provider initializes with defaults."""
        provider = PhilPapersProvider({"type": "philpapers"})

        assert provider.min_interval_seconds == 2.0
        assert provider.snippet_max_chars == 700
        assert provider.timeout == 15
        assert provider.api_id is None
        assert provider.api_key is None
        assert provider.subject_categories == []
        assert provider.include_books is True
        assert provider.open_access_only is False

    def test_provider_initialization_with_config(self):
        """Test provider initializes with custom config."""
        provider = PhilPapersProvider({
            "type": "philpapers",
            "api_id": "test-id",
            "api_key": "test-key",
            "timeout": 30,
            "min_interval_seconds": 3.0,
            "subject_categories": ["ethics", "epistemology"],
            "include_books": False,
            "open_access_only": True,
        })

        assert provider.api_id == "test-id"
        assert provider.api_key == "test-key"
        assert provider.timeout == 30
        assert provider.min_interval_seconds == 3.0
        assert provider.subject_categories == ["ethics", "epistemology"]
        assert provider.include_books is False
        assert provider.open_access_only is True

    def test_is_available_always_true(self):
        """Test provider is always available (OAI-PMH is free)."""
        provider = PhilPapersProvider({})
        assert provider.is_available() is True

    def test_validate_config_valid_categories(self):
        """Test validation passes for valid categories."""
        provider = PhilPapersProvider({
            "subject_categories": ["ethics", "epistemology", "metaphysics"]
        })
        provider.validate_config()  # Should not raise

    def test_validate_config_invalid_category(self):
        """Test validation fails for invalid category."""
        provider = PhilPapersProvider({
            "subject_categories": ["invalid_category"]
        })
        with pytest.raises(ValueError, match="Invalid subject categories"):
            provider.validate_config()

    def test_get_name(self):
        """Test get_name returns correct value."""
        provider = PhilPapersProvider({})
        assert provider.get_name() == "PhilPapers"

    def test_name_property(self):
        """Test name property returns correct value."""
        provider = PhilPapersProvider({})
        assert provider.name == "philpapers"

    def test_list_subject_categories(self):
        """Test listing subject categories."""
        provider = PhilPapersProvider({})
        categories = provider.list_subject_categories()

        assert "epistemology" in categories
        assert "ethics" in categories
        assert "metaphysics" in categories
        assert categories["epistemology"] == "Epistemology"

    @patch("requests.get")
    def test_search_returns_results(self, mock_get):
        """Test search returns parsed results matching query terms."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({"type": "philpapers"})
        # Search for terms that match the first result
        results = provider.search("knowledge epistemology", max_results=5)

        # Should return 1 result matching "knowledge" or "epistemology"
        assert len(results) >= 1

        # Check first result
        assert results[0].title == "The Problem of Knowledge in Epistemology"
        assert "philpapers.org" in results[0].url
        assert "fundamental nature of knowledge" in results[0].snippet
        assert results[0].metadata["philpapers_id"] == "SMIETH-1"
        assert results[0].metadata["authors"] == ["John Smith", "Jane Doe"]
        assert results[0].metadata["year"] == 2023
        assert "Epistemology" in results[0].metadata["subjects"]
        assert "citation" in results[0].metadata

    @patch("requests.get")
    def test_search_returns_multiple_results(self, mock_get):
        """Test search returns all results when query is empty/subject-only."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({"type": "philpapers"})
        # Search with subject filter only - text is empty so all records pass
        results = provider.search("subject:ethics", max_results=5)

        # With no text filter, all records should be returned
        assert len(results) == 2

        # Check both results are returned
        titles = [r.title for r in results]
        assert "The Problem of Knowledge in Epistemology" in titles
        assert "Virtue Ethics and Moral Character" in titles

    @patch("requests.get")
    def test_search_empty_query_returns_empty(self, mock_get):
        """Test empty query returns empty results."""
        provider = PhilPapersProvider({})
        results = provider.search("", max_results=5)

        assert results == []
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_search_handles_api_error(self, mock_get):
        """Test search handles API errors gracefully."""
        mock_get.side_effect = Exception("API Error")

        provider = PhilPapersProvider({})
        results = provider.search("test query", max_results=5)

        assert results == []

    @patch("requests.get")
    def test_search_handles_oai_error(self, mock_get):
        """Test search handles OAI-PMH errors gracefully."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_ERROR_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        results = provider.search("nonexistent query", max_results=5)

        # Should return empty list, not raise exception
        assert results == []

    @patch("requests.get")
    def test_search_with_author_filter(self, mock_get):
        """Test search with author: prefix filters results."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        results = provider.search("author:Smith", max_results=5)

        # Should only return results with Smith as author
        for result in results:
            authors_lower = [a.lower() for a in result.metadata["authors"]]
            assert any("smith" in a for a in authors_lower)

    @patch("requests.get")
    def test_search_with_subject_filter(self, mock_get):
        """Test search with subject: prefix."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        results = provider.search("subject:ethics virtue", max_results=5)

        # Verify set parameter was used
        call_args = mock_get.call_args
        # The subject should be included in the params
        assert mock_get.called

    @patch("requests.get")
    def test_search_by_category(self, mock_get):
        """Test searching by category."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        results = provider.search_by_category("ethics", max_results=5)

        assert mock_get.called

    def test_format_authors_single(self):
        """Test author formatting with single author."""
        provider = PhilPapersProvider({})
        assert provider._format_authors(["John Doe"]) == "John Doe"

    def test_format_authors_three(self):
        """Test author formatting with three authors."""
        provider = PhilPapersProvider({})
        result = provider._format_authors(["A", "B", "C"])
        assert result == "A, B, C"

    def test_format_authors_more_than_three(self):
        """Test author formatting with more than three authors."""
        provider = PhilPapersProvider({})
        result = provider._format_authors(["A", "B", "C", "D", "E"])
        assert result == "A, B, C, et al."

    def test_format_authors_empty(self):
        """Test author formatting with no authors."""
        provider = PhilPapersProvider({})
        assert provider._format_authors([]) == "Unknown"

    def test_truncate_short_text(self):
        """Test truncation of short text."""
        provider = PhilPapersProvider({"snippet_max_chars": 100})
        text = "Short text"
        assert provider._truncate(text) == text

    def test_truncate_long_text(self):
        """Test truncation of long text."""
        provider = PhilPapersProvider({"snippet_max_chars": 20})
        text = "This is a very long text that should be truncated"
        result = provider._truncate(text)
        assert len(result) <= 23  # 20 + "..."
        assert result.endswith("...")

    def test_extract_year_valid(self):
        """Test year extraction from date strings."""
        provider = PhilPapersProvider({})
        assert provider._extract_year("2023") == 2023
        assert provider._extract_year("2023-06-15") == 2023
        assert provider._extract_year("Published in 2021") == 2021

    def test_extract_year_invalid(self):
        """Test year extraction with invalid input."""
        provider = PhilPapersProvider({})
        assert provider._extract_year("") is None
        assert provider._extract_year("no year here") is None

    def test_parse_query_plain_text(self):
        """Test query parsing with plain text."""
        provider = PhilPapersProvider({})
        parsed = provider._parse_query("knowledge justified belief")

        assert parsed["text"] == "knowledge justified belief"
        assert parsed["author"] is None
        assert parsed["subject"] is None

    def test_parse_query_with_author(self):
        """Test query parsing with author prefix."""
        provider = PhilPapersProvider({})
        parsed = provider._parse_query("author:Smith epistemology")

        assert "epistemology" in parsed["text"]
        assert parsed["author"] == "Smith"

    def test_parse_query_with_subject(self):
        """Test query parsing with subject prefix."""
        provider = PhilPapersProvider({})
        parsed = provider._parse_query("subject:ethics moral philosophy")

        assert "moral philosophy" in parsed["text"]
        assert parsed["subject"] == "ethics"

    @patch("requests.get")
    def test_score_calculation(self, mock_get):
        """Test score calculation is position-based."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        results = provider.search("test", max_results=5)

        # All scores should be between 0.2 and 1.0
        for result in results:
            assert 0.2 <= result.score <= 1.0

        # First result should have highest score
        if len(results) > 1:
            assert results[0].score > results[-1].score

    @patch("requests.get")
    def test_get_paper_by_id(self, mock_get):
        """Test fetching paper by PhilPapers ID."""
        response = Mock()
        response.status_code = 200
        response.content = OAI_PMH_RESPONSE.encode("utf-8")
        mock_get.return_value = response

        provider = PhilPapersProvider({})
        # This would need a GetRecord response, but we test the method is callable
        result = provider.get_paper_by_id("SMIETH-1")

        # Since our mock returns ListRecords format, this may return None
        # but the method should not raise
        assert mock_get.called
