"""
CalDAV provider for self-hosted calendars.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.providers.communication.models import CalendarEvent

logger = get_logger(__name__)


class CalDAVProvider(BaseRemoteProvider):
    """
    CalDAV provider for reading and mutating events.
    """

    DEFAULT_TIMEOUT = 20

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.calendar_url = config.get("calendar_url") or config.get("base_url")
        self.username = config.get("username") or config.get("user")
        self.password = config.get("password") or config.get("pass")
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT

    @property
    def name(self) -> str:
        return "caldav"

    def get_name(self) -> str:
        return "CalDAV"

    def get_description(self) -> str:
        return "CalDAV provider for self-hosted calendars."

    def validate_config(self) -> None:
        if not self.calendar_url:
            raise ValueError("CalDAV calendar_url is required")
        if not self.username or not self.password:
            raise ValueError("CalDAV username and password are required")

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
        if time_min is None:
            time_min = datetime.now(timezone.utc) - timedelta(days=1)
        if time_max is None:
            time_max = datetime.now(timezone.utc) + timedelta(days=30)
        report = _build_report(time_min, time_max)
        response = self._request("REPORT", self.calendar_url, data=report)
        events = _parse_ics_events(response, query=query)
        return events[:max_results]

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        event_id = event.event_id or str(uuid4())
        ics = _build_ics(event_id, event)
        url = self._event_url(event_id)
        self._request("PUT", url, data=ics, content_type="text/calendar")
        event.event_id = event_id
        return event

    def update_event(self, event_id: str, updates: Dict[str, Any]) -> CalendarEvent:
        event = _event_from_updates(event_id, updates)
        ics = _build_ics(event_id, event)
        url = self._event_url(event_id)
        self._request("PUT", url, data=ics, content_type="text/calendar")
        return event

    def delete_event(self, event_id: str) -> None:
        url = self._event_url(event_id)
        self._request("DELETE", url)

    def _event_url(self, event_id: str) -> str:
        base = str(self.calendar_url).rstrip("/")
        return f"{base}/{event_id}.ics"

    def _request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        headers = {"Depth": "1"}
        if content_type:
            headers["Content-Type"] = content_type
        response = requests.request(
            method,
            url,
            data=data,
            headers=headers,
            auth=(self.username, self.password),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text or ""


def _build_report(time_min: datetime, time_max: datetime) -> str:
    start = time_min.strftime("%Y%m%dT%H%M%SZ")
    end = time_max.strftime("%Y%m%dT%H%M%SZ")
    return f"""<?xml version="1.0" encoding="utf-8" ?>
<c:calendar-query xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop xmlns:d="DAV:">
    <d:getetag />
    <c:calendar-data />
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:time-range start="{start}" end="{end}" />
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>
"""


def _parse_ics_events(text: str, *, query: Optional[str]) -> List[CalendarEvent]:
    events: List[CalendarEvent] = []
    blocks = text.split("BEGIN:VEVENT")
    for block in blocks[1:]:
        body = block.split("END:VEVENT")[0]
        event = _parse_ics_block(body)
        if event is None:
            continue
        if query:
            combined = f"{event.title} {event.description or ''}".lower()
            if query.lower() not in combined:
                continue
        events.append(event)
    return events


def _parse_ics_block(body: str) -> Optional[CalendarEvent]:
    fields: Dict[str, str] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.upper()] = value
    uid = fields.get("UID") or ""
    if not uid:
        return None
    start = _parse_ics_datetime(fields.get("DTSTART"))
    end = _parse_ics_datetime(fields.get("DTEND"))
    attendees = [
        value.replace("mailto:", "")
        for key, value in fields.items()
        if key.startswith("ATTENDEE")
    ]
    return CalendarEvent(
        event_id=uid,
        title=fields.get("SUMMARY", ""),
        start=start,
        end=end,
        location=fields.get("LOCATION"),
        description=fields.get("DESCRIPTION"),
        attendees=attendees,
        status=fields.get("STATUS"),
        metadata={"provider": "caldav", "source_id": uid},
    )


def _parse_ics_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _build_ics(event_id: str, event: CalendarEvent) -> str:
    start = event.start or datetime.now(timezone.utc)
    end = event.end or (start + timedelta(hours=1))
    return (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//Local Second Mind//EN\n"
        "BEGIN:VEVENT\n"
        f"UID:{event_id}\n"
        f"DTSTART:{start.strftime('%Y%m%dT%H%M%SZ')}\n"
        f"DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}\n"
        f"SUMMARY:{event.title}\n"
        + (f"DESCRIPTION:{event.description}\n" if event.description else "")
        + (f"LOCATION:{event.location}\n" if event.location else "")
        + "END:VEVENT\n"
        "END:VCALENDAR\n"
    )


def _event_from_updates(event_id: str, updates: Dict[str, Any]) -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title=str(updates.get("summary") or updates.get("title") or ""),
        start=updates.get("start"),
        end=updates.get("end"),
        location=updates.get("location"),
        description=updates.get("description"),
        attendees=updates.get("attendees") or [],
        status=updates.get("status"),
        metadata={"provider": "caldav", "source_id": event_id},
    )


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
