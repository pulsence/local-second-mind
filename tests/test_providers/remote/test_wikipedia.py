"""
Tests for Wikipedia provider implementation.
"""

from unittest.mock import Mock, patch

from lsm.query.remote.wikipedia import WikipediaProvider


class TestWikipediaProvider:
    """Tests for Wikipedia provider."""

    def test_wikipedia_provider_initialization_defaults(self):
        """Test Wikipedia provider initializes with defaults."""
        provider = WikipediaProvider({"type": "wikipedia"})

        assert provider.language == "en"
        assert provider.endpoint.startswith("https://en.wikipedia.org/")
        assert provider.min_interval_seconds == 1.0
        assert provider.section_limit == 2
        assert provider.snippet_max_chars == 600

    @patch("requests.get")
    def test_wikipedia_search_success(self, mock_get):
        """Test Wikipedia search returns parsed results."""
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "query": {
                "search": [
                    {"title": "Python (programming language)"}
                ]
            }
        }

        details_response = Mock()
        details_response.status_code = 200
        details_response.json.return_value = {
            "query": {
                "pages": {
                    "123": {
                        "pageid": 123,
                        "title": "Python (programming language)",
                        "extract": "Python is a language.\n==History==\nHistory text.",
                        "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "touched": "2025-01-01T00:00:00Z",
                    }
                }
            }
        }

        mock_get.side_effect = [search_response, details_response]

        provider = WikipediaProvider({"type": "wikipedia", "user_agent": "TestAgent"})
        results = provider.search("python history", max_results=1)

        assert len(results) == 1
        assert results[0].title == "Python (programming language)"
        assert "History" in results[0].snippet
        assert results[0].metadata["is_disambiguation"] is False
        assert "citation" in results[0].metadata

    @patch("requests.get")
    def test_wikipedia_skips_disambiguation(self, mock_get):
        """Test Wikipedia skips disambiguation pages by default."""
        search_response = Mock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "query": {
                "search": [
                    {"title": "Mercury"},
                    {"title": "Mercury (planet)"},
                ]
            }
        }

        details_response = Mock()
        details_response.status_code = 200
        details_response.json.return_value = {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "title": "Mercury",
                        "extract": "May refer to many things.",
                        "pageprops": {"disambiguation": ""},
                        "fullurl": "https://en.wikipedia.org/wiki/Mercury",
                    },
                    "2": {
                        "pageid": 2,
                        "title": "Mercury (planet)",
                        "extract": "Mercury is the smallest planet.",
                        "fullurl": "https://en.wikipedia.org/wiki/Mercury_(planet)",
                    },
                }
            }
        }

        mock_get.side_effect = [search_response, details_response]

        provider = WikipediaProvider({"type": "wikipedia", "user_agent": "TestAgent"})
        results = provider.search("Mercury", max_results=1)

        assert len(results) == 1
        assert results[0].title == "Mercury (planet)"

    def test_wikipedia_rate_limiting(self):
        """Test Wikipedia rate limiting sleeps when needed."""
        provider = WikipediaProvider(
            {"type": "wikipedia", "user_agent": "TestAgent", "min_interval_seconds": 1.0}
        )
        provider._last_request_time = 10.0

        with patch("lsm.query.remote.wikipedia.time.time") as mock_time, \
            patch("lsm.query.remote.wikipedia.time.sleep") as mock_sleep:
            mock_time.side_effect = [10.2, 11.0]

            provider._throttle()

            mock_sleep.assert_called_once()
            sleep_seconds = mock_sleep.call_args[0][0]
            assert round(sleep_seconds, 2) == 0.8
