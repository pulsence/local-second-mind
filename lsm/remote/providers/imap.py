"""Backward-compatible import for IMAP provider."""

from __future__ import annotations

from lsm.remote.providers.communication.imap import IMAPProvider

__all__ = ["IMAPProvider"]
