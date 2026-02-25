"""
Google Calendar provider via Google Calendar API.
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


class GoogleCalendarProvider(BaseRemoteProvider):
    """
    Google Calendar provider using OAuth2.
    """

    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    API_BASE = "https://www.googleapis.com/calendar/v3"
    DEFAULT_TIMEOUT = 20
    DEFAULT_SCOPES = [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ]

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.endpoint = config.get("endpoint") or self.API_BASE
        self.calendar_id = config.get("calendar_id") or "primary"
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        oauth = config.get("oauth") or {}
        scopes = oauth.get("scopes") if isinstance(oauth, dict) else None
        if not scopes:
            oauth = dict(oauth) if isinstance(oauth, dict) else {}
            oauth["scopes"] = list(self.DEFAULT_SCOPES)
        self.oauth_config = _coerce_oauth(oauth)
        token_store = OAuthTokenStore(_oauth_root(config))
        self.oauth_client = OAuth2Client(
            provider_name="google_calendar",
            oauth_config=self.oauth_config,
            auth_url=self.AUTH_URL,
            token_url=self.TOKEN_URL,
            token_store=token_store,
            extra_auth_params={"access_type": "offline", "prompt": "consent"},
        )

    @property
    def name(self) -> str:
        return "google_calendar"

    def get_name(self) -> str:
        return "Google Calendar"

    def get_description(self) -> str:
        return "Google Calendar events via OAuth2."

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
            "maxResults": max_results,
            "singleEvents": "true",
            "orderBy": "startTime",
        }
        if query:
            params["q"] = query
        if time_min:
            params["timeMin"] = time_min.isoformat()
        if time_max:
            params["timeMax"] = time_max.isoformat()
        url = f"{self.endpoint}/calendars/{self.calendar_id}/events"
        data = self._request(url, params=params)
        items = data.get("items", []) if isinstance(data, dict) else []
        events: List[CalendarEvent] = []
        for item in items[:max_results]:
            events.append(_parse_event(item))
        return events

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        url = f"{self.endpoint}/calendars/{self.calendar_id}/events"
        payload = _event_payload(event)
        data = self._request(url, method="POST", json_data=payload)
        return _parse_event(data)

    def update_event(self, event_id: str, updates: Dict[str, Any]) -> CalendarEvent:
        url = f"{self.endpoint}/calendars/{self.calendar_id}/events/{event_id}"
        data = self._request(url, method="PATCH", json_data=updates)
        return _parse_event(data)

    def delete_event(self, event_id: str) -> None:
        url = f"{self.endpoint}/calendars/{self.calendar_id}/events/{event_id}"
        self._request(url, method="DELETE", json_data=None)

    def _headers(self) -> Dict[str, str]:
        token = self.oauth_client.get_access_token(allow_interactive=True)
        return {"Authorization": f"Bearer {token}"}

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
        title=str(payload.get("summary") or ""),
        start=_parse_datetime(start.get("dateTime") or start.get("date")),
        end=_parse_datetime(end.get("dateTime") or end.get("date")),
        location=payload.get("location"),
        description=payload.get("description"),
        attendees=[
            attendee.get("email")
            for attendee in payload.get("attendees") or []
            if attendee.get("email")
        ],
        status=payload.get("status"),
        metadata={"provider": "google_calendar", "source_id": payload.get("id")},
    )


def _event_payload(event: CalendarEvent) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "summary": event.title,
        "description": event.description,
        "location": event.location,
    }
    if event.start:
        payload["start"] = {"dateTime": event.start.isoformat()}
    if event.end:
        payload["end"] = {"dateTime": event.end.isoformat()}
    if event.attendees:
        payload["attendees"] = [{"email": email} for email in event.attendees]
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
