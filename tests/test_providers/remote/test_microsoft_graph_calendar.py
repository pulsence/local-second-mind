from __future__ import annotations

from unittest.mock import Mock, patch

from lsm.remote.providers.communication.microsoft_graph_calendar import (
    MicrosoftGraphCalendarProvider,
)
from lsm.remote.providers.communication.models import CalendarEvent
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


def _mock_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.content = b"{}"
    response.raise_for_status = lambda: None
    return response


class TestMicrosoftGraphCalendarProvider(RemoteProviderOutputTest):
    @patch(
        "lsm.remote.providers.communication.microsoft_graph_calendar.OAuth2Client.get_access_token",
        return_value="token",
    )
    @patch("requests.request")
    def test_search_returns_results(self, mock_request: Mock, _mock_token: Mock):
        mock_request.return_value = _mock_response(
            {
                "value": [
                    {
                        "id": "evt1",
                        "subject": "Meeting",
                        "start": {"dateTime": "2024-01-01T10:00:00Z"},
                        "end": {"dateTime": "2024-01-01T11:00:00Z"},
                    }
                ]
            }
        )
        provider = MicrosoftGraphCalendarProvider(
            {
                "oauth": {
                    "client_id": "client",
                    "client_secret": "secret",
                    "scopes": ["Calendars.Read"],
                }
            }
        )
        results = provider.search("meeting", max_results=1)
        assert len(results) == 1
        assert results[0].metadata["event_id"] == "evt1"
        self.assert_valid_output(results)
        assert mock_request.call_args.kwargs["headers"]["Authorization"] == "Bearer token"

    @patch(
        "lsm.remote.providers.communication.microsoft_graph_calendar.OAuth2Client.get_access_token",
        return_value="token",
    )
    @patch("requests.request")
    def test_create_event(self, mock_request: Mock, _mock_token: Mock):
        mock_request.return_value = _mock_response(
            {
                "id": "evt1",
                "subject": "Meeting",
                "start": {"dateTime": "2024-01-01T10:00:00Z"},
                "end": {"dateTime": "2024-01-01T11:00:00Z"},
            }
        )
        provider = MicrosoftGraphCalendarProvider(
            {
                "oauth": {
                    "client_id": "client",
                    "client_secret": "secret",
                    "scopes": ["Calendars.ReadWrite"],
                }
            }
        )
        event = CalendarEvent(event_id="", title="Meeting", start=None, end=None)
        created = provider.create_event(event)
        assert created.event_id == "evt1"
