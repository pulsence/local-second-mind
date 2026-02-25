"""Backward-compatible import for Gmail provider."""

from __future__ import annotations

from lsm.remote.providers.communication.gmail import GmailProvider

__all__ = ["GmailProvider"]
