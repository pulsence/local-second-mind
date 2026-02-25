from __future__ import annotations

from unittest.mock import Mock, patch

from lsm.remote.providers.communication.caldav import CalDAVProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


def _mock_response(text: str) -> Mock:
    response = Mock()
    response.status_code = 207
    response.text = text
    response.raise_for_status = lambda: None
    return response


class TestCalDAVProvider(RemoteProviderOutputTest):
    @patch("requests.request")
    def test_search_returns_results(self, mock_request: Mock):
        mock_request.return_value = _mock_response(
            "BEGIN:VEVENT\n"
            "UID:evt1\n"
            "SUMMARY:Meeting\n"
            "DTSTART:20240101T100000Z\n"
            "DTEND:20240101T110000Z\n"
            "END:VEVENT\n"
        )
        provider = CalDAVProvider(
            {
                "calendar_url": "https://caldav.example.com/cal",
                "username": "user",
                "password": "pass",
            }
        )
        results = provider.search("meeting", max_results=1)
        assert len(results) == 1
        assert results[0].metadata["event_id"] == "evt1"
        self.assert_valid_output(results)
