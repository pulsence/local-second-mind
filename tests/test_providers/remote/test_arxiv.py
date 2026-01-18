"""
Tests for arXiv provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.remote.providers.arxiv import ArXivProvider


ARXIV_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Test Paper</title>
    <summary>This is an abstract for testing.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <link rel="alternate" type="text/html" href="http://arxiv.org/abs/1234.5678v1"/>
    <link title="pdf" rel="related" type="application/pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
    <category term="cs.AI"/>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>
"""


class TestArXivProvider:
    """Tests for arXiv provider."""

    def test_arxiv_provider_initialization_defaults(self):
        """Test arXiv provider initializes with defaults."""
        provider = ArXivProvider({"type": "arxiv"})

        assert provider.endpoint == "https://export.arxiv.org/api/query"
        assert provider.min_interval_seconds == 3.0
        assert provider.snippet_max_chars == 700
        assert provider.sort_by == "relevance"
        assert provider.sort_order == "descending"

    def test_build_search_query_fielded(self):
        """Test arXiv query parsing for fielded syntax."""
        provider = ArXivProvider({"type": "arxiv"})
        search_query = provider._build_search_query("title:graph nets author:kipf")

        assert "ti:\"graph nets\"" in search_query
        assert "au:kipf" in search_query

    def test_build_search_query_embedded_field(self):
        """Test arXiv query parsing with embedded field clause."""
        provider = ArXivProvider({"type": "arxiv"})
        search_query = provider._build_search_query('What about articles on "title:pulsar timing"?')

        assert "ti:\"pulsar timing\"" in search_query

    @patch("requests.get")
    def test_arxiv_search_parses_results(self, mock_get):
        """Test arXiv search returns parsed results."""
        response = Mock()
        response.status_code = 200
        response.text = ARXIV_FEED_XML
        mock_get.return_value = response

        provider = ArXivProvider({"type": "arxiv", "user_agent": "TestAgent"})
        results = provider.search("graph neural networks", max_results=1)

        assert len(results) == 1
        assert results[0].title == "Test Paper"
        assert results[0].url == "https://arxiv.org/abs/1234.5678v1"
        assert results[0].metadata["pdf_url"] == "https://arxiv.org/pdf/1234.5678v1"
        assert results[0].metadata["primary_category"] == "cs.AI"
        assert "citation" in results[0].metadata
