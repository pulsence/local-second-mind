"""
Microsoft Graph Calendar provider.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from lsm.config.models import OAuthConfig
from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.oauth import OAuth2Client, OAuthTokenStore
from lsm.remote.providers.communication.models import CalendarEvent

logger = get_logger(__name__)


class MicrosoftGraphCalendarProvider(BaseRemoteProvider):
    """
    Outlook calendar provider via Microsoft Graph API.
    """

    AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
    TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    API_BASE = "https://graph.microsoft.com/v1.0"
    DEFAULT_TIMEOUT = 20
    DEFAULT_SCOPES = [
        "offline_access",
        "Calendars.Read",
        "Calendars.ReadWrite",
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint") or self.API_BASE
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        oauth = config.get("oauth") or {}
        scopes = oauth.get("scopes") if isinstance(oauth, dict) else None
        if not scopes:
            oauth = dict(oauth) if isinstance(oauth, dict) else {}
            oauth["scopes"] = list(self.DEFAULT_SCOPES)
        self.oauth_config = _coerce_oauth(oauth)
        token_store = OAuthTokenStore(_oauth_root(config))
        self.oauth_client = OAuth2Client(
            provider_name="microsoft_graph_calendar",
            oauth_config=self.oauth_config,
            auth_url=self.AUTH_URL,
            token_url=self.TOKEN_URL,
            token_store=token_store,
        )

    @property
    def name(self) -> str:
        return "microsoft_graph_calendar"

    def get_name(self) -> str:
        return "Microsoft Graph Calendar"

    def get_description(self) -> str:
        return "Outlook calendar events via Microsoft Graph API."

    def validate_config(self) -> None:
        self.oauth_config.validate()

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Search query.", "required": False},
            {"name": "time_min", "type": "string", "description": "ISO start time.", "required": False},
            {"name": "time_max", "type": "string", "description": "ISO end time.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "event_id", "type": "string", "description": "Event ID."},
            {"name": "start", "type": "string", "description": "Event start time."},
            {"name": "end", "type": "string", "description": "Event end time."},
            {"name": "location", "type": "string", "description": "Event location."},
            {"name": "attendees", "type": "array[string]", "description": "Attendee list."},
            {"name": "status", "type": "string", "description": "Event status."},
        ]

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        events = self.list_events(query=query, max_results=max_results)
        return [_event_to_result(event) for event in events]

    def list_events(
        self,
        *,
        query: Optional[str] = None,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
        max_results: int = 10,
    ) -> List[CalendarEvent]:
        params: Dict[str, Any] = {
            "$top": max_results,
            "$orderby": "start/dateTime",
        }
        url = f"{self.endpoint}/me/events"
        if time_min and time_max:
            url = f"{self.endpoint}/me/calendarView"
            params["startDateTime"] = time_min.isoformat()
            params["endDateTime"] = time_max.isoformat()
        if query:
            params["$search"] = f"\"{query}\""
        data = self._request(url, params=params)
        items = data.get("value", []) if isinstance(data, dict) else []
        return [_parse_event(item) for item in items[:max_results]]

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        url = f"{self.endpoint}/me/events"
        payload = _event_payload(event)
        data = self._request(url, method="POST", json_data=payload)
        return _parse_event(data)

    def update_event(self, event_id: str, updates: Dict[str, Any]) -> CalendarEvent:
        url = f"{self.endpoint}/me/events/{event_id}"
        data = self._request(url, method="PATCH", json_data=updates)
        return _parse_event(data)

    def delete_event(self, event_id: str) -> None:
        url = f"{self.endpoint}/me/events/{event_id}"
        self._request(url, method="DELETE", json_data=None)

    def _headers(self) -> Dict[str, str]:
        token = self.oauth_client.get_access_token(allow_interactive=True)
        return {
            "Authorization": f"Bearer {token}",
            "ConsistencyLevel": "eventual",
        }

    def _request(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_data,
            headers=self._headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json() if response.content else {}
        return data if isinstance(data, dict) else {}


def _parse_event(payload: Dict[str, Any]) -> CalendarEvent:
    start = payload.get("start", {})
    end = payload.get("end", {})
    return CalendarEvent(
        event_id=str(payload.get("id") or ""),
        title=str(payload.get("subject") or ""),
        start=_parse_datetime(start.get("dateTime")),
        end=_parse_datetime(end.get("dateTime")),
        location=(payload.get("location") or {}).get("displayName"),
        description=payload.get("bodyPreview"),
        attendees=[
            attendee.get("emailAddress", {}).get("address")
            for attendee in payload.get("attendees") or []
            if attendee.get("emailAddress", {}).get("address")
        ],
        status=payload.get("showAs"),
        metadata={"provider": "microsoft_graph_calendar", "source_id": payload.get("id")},
    )


def _event_payload(event: CalendarEvent) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "subject": event.title,
        "body": {"contentType": "Text", "content": event.description or ""},
        "location": {"displayName": event.location} if event.location else None,
    }
    if event.start:
        payload["start"] = {"dateTime": event.start.isoformat(), "timeZone": "UTC"}
    if event.end:
        payload["end"] = {"dateTime": event.end.isoformat(), "timeZone": "UTC"}
    if event.attendees:
        payload["attendees"] = [
            {"emailAddress": {"address": email}, "type": "required"}
            for email in event.attendees
        ]
    payload = {k: v for k, v in payload.items() if v is not None}
    return payload


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_to_result(event: CalendarEvent) -> RemoteResult:
    return RemoteResult(
        title=event.title or event.event_id,
        url=event.event_id,
        snippet=event.description or event.title or "",
        score=1.0,
        metadata={
            "event_id": event.event_id,
            "start": event.start.isoformat() if event.start else None,
            "end": event.end.isoformat() if event.end else None,
            "location": event.location,
            "attendees": event.attendees,
            "status": event.status,
            "source_id": event.event_id,
        },
    )


def _coerce_oauth(raw: Dict[str, Any]):
    if isinstance(raw, OAuthConfig):
        return raw
    if not isinstance(raw, dict):
        return OAuthConfig(client_id="", client_secret="", scopes=[])
    return OAuthConfig(
        client_id=str(raw.get("client_id", "")).strip(),
        client_secret=str(raw.get("client_secret", "")).strip(),
        scopes=list(raw.get("scopes") or []),
        redirect_uri=str(raw.get("redirect_uri", OAuthConfig.redirect_uri)).strip(),
        refresh_buffer_seconds=int(
            raw.get("refresh_buffer_seconds", OAuthConfig.refresh_buffer_seconds)
        ),
    )


def _oauth_root(config: Dict[str, Any]) -> Optional[str]:
    value = config.get("global_folder")
    if not value:
        return None
    return str(Path(value) / "oauth_tokens")
