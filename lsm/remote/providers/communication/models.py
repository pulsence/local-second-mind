"""
Shared models for communication providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from lsm.remote.base import RemoteResult


@dataclass
class EmailMessage:
    message_id: str
    subject: str
    sender: str
    recipients: List[str] = field(default_factory=list)
    snippet: str = ""
    received_at: Optional[datetime] = None
    thread_id: Optional[str] = None
    is_unread: bool = False
    folder: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_remote_result(self) -> RemoteResult:
        title = self.subject or self.snippet or self.message_id
        url = self.metadata.get("url") or self.message_id
        summary = self.snippet or title
        payload = {
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "from": self.sender,
            "to": self.recipients,
            "received_at": self.received_at.isoformat() if self.received_at else None,
            "is_unread": self.is_unread,
            "folder": self.folder,
            "labels": list(self.labels or []),
            "source_id": self.message_id,
        }
        payload.update(self.metadata or {})
        return RemoteResult(
            title=title,
            url=str(url),
            snippet=summary,
            score=1.0,
            metadata=payload,
        )


@dataclass
class EmailDraft:
    draft_id: str
    subject: str
    recipients: List[str]
    body: str
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
