"""Backward-compatible import for Microsoft Graph Calendar provider."""

from __future__ import annotations

from lsm.remote.providers.communication.microsoft_graph_calendar import (
    MicrosoftGraphCalendarProvider,
)

__all__ = ["MicrosoftGraphCalendarProvider"]
