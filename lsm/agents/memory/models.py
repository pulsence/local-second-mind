"""
Datamodels for agent memory storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import uuid4

VALID_MEMORY_TYPES = {"pinned", "project_fact", "task_state", "cache"}
VALID_MEMORY_SCOPES = {"global", "agent", "project"}
VALID_CANDIDATE_STATUSES = {"pending", "promoted", "rejected"}


def now_utc() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    """
    Persisted memory unit.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    type: str = "project_fact"
    key: str = ""
    value: Any = None
    scope: str = "project"
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=now_utc)
    last_used_at: datetime = field(default_factory=now_utc)
    expires_at: Optional[datetime] = None
    source_run_id: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id).strip() or str(uuid4())
        self.type = str(self.type).strip().lower()
        self.key = str(self.key).strip()
        self.scope = str(self.scope).strip().lower()
        self.tags = [str(tag).strip() for tag in self.tags if str(tag).strip()]
        self.tags = list(dict.fromkeys(self.tags))
        self.confidence = float(self.confidence)
        self.created_at = _as_utc(self.created_at)
        self.last_used_at = _as_utc(self.last_used_at)
        if self.expires_at is not None:
            self.expires_at = _as_utc(self.expires_at)
        self.source_run_id = str(self.source_run_id).strip()

    def validate(self) -> None:
        """Validate memory fields."""
        if self.type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"memory.type must be one of {sorted(VALID_MEMORY_TYPES)}"
            )
        if self.scope not in VALID_MEMORY_SCOPES:
            raise ValueError(
                f"memory.scope must be one of {sorted(VALID_MEMORY_SCOPES)}"
            )
        if not self.key:
            raise ValueError("memory.key cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("memory.confidence must be between 0.0 and 1.0")


@dataclass
class MemoryCandidate:
    """
    Candidate memory awaiting lifecycle decision.
    """

    id: str
    memory: Memory
    provenance: str
    rationale: str
    status: str = "pending"

    def __post_init__(self) -> None:
        self.id = str(self.id).strip() or str(uuid4())
        self.provenance = str(self.provenance).strip()
        self.rationale = str(self.rationale).strip()
        self.status = str(self.status).strip().lower()

    def validate(self) -> None:
        """Validate candidate fields."""
        self.memory.validate()
        if not self.provenance:
            raise ValueError("candidate.provenance cannot be empty")
        if not self.rationale:
            raise ValueError("candidate.rationale cannot be empty")
        if self.status not in VALID_CANDIDATE_STATUSES:
            raise ValueError(
                "candidate.status must be one of "
                f"{sorted(VALID_CANDIDATE_STATUSES)}"
            )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
