from __future__ import annotations

from unittest.mock import Mock, patch

from lsm.remote.providers.communication.gmail import GmailProvider
from tests.test_providers.remote.test_base import RemoteProviderOutputTest


def _mock_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = lambda: None
    return response


class TestGmailProvider(RemoteProviderOutputTest):
    @patch("lsm.remote.providers.communication.gmail.OAuth2Client.get_access_token", return_value="token")
    @patch("requests.request")
    def test_search_returns_results(self, mock_request: Mock, _mock_token: Mock):
        mock_request.side_effect = [
            _mock_response({"messages": [{"id": "msg1", "threadId": "t1"}]}),
            _mock_response(
                {
                    "id": "msg1",
                    "threadId": "t1",
                    "snippet": "Hello",
                    "labelIds": ["UNREAD"],
                    "payload": {
                        "headers": [
                            {"name": "Subject", "value": "Test"},
                            {"name": "From", "value": "sender@example.com"},
                            {"name": "To", "value": "me@example.com"},
                        ]
                    },
                }
            ),
        ]

        provider = GmailProvider(
            {
                "oauth": {
                    "client_id": "client",
                    "client_secret": "secret",
                    "scopes": ["scope"],
                }
            }
        )
        results = provider.search("hello", max_results=1)
        assert len(results) == 1
        assert results[0].metadata["message_id"] == "msg1"
        self.assert_valid_output(results)

        assert mock_request.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer token"

    @patch("lsm.remote.providers.communication.gmail.OAuth2Client.get_access_token", return_value="token")
    @patch("requests.request")
    def test_create_draft(self, mock_request: Mock, _mock_token: Mock):
        mock_request.return_value = _mock_response({"id": "draft1"})
        provider = GmailProvider(
            {
                "oauth": {
                    "client_id": "client",
                    "client_secret": "secret",
                    "scopes": ["scope"],
                }
            }
        )
        draft = provider.create_draft(
            recipients=["me@example.com"],
            subject="Subject",
            body="Body",
        )
        assert draft.draft_id == "draft1"
        payload = mock_request.call_args.kwargs["json"]
        assert "raw" in payload["message"]
