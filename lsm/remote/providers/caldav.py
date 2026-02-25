"""Backward-compatible import for CalDAV provider."""

from __future__ import annotations

from lsm.remote.providers.communication.caldav import CalDAVProvider

__all__ = ["CalDAVProvider"]
