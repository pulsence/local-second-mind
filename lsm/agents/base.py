"""
Base classes and state models for agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional

from .models import AgentContext, AgentLogEntry


class AgentStatus(str, Enum):
    """
    Lifecycle status for an agent run.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentState:
    """
    Mutable state for an agent instance.
    """

    status: AgentStatus = AgentStatus.IDLE
    """Current lifecycle status."""

    current_task: Optional[str] = None
    """Current task description."""

    log_entries: List[AgentLogEntry] = field(default_factory=list)
    """Chronological log entries for this run."""

    artifacts: List[str] = field(default_factory=list)
    """Tracked artifact paths produced during this run."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """Creation timestamp in UTC."""

    updated_at: datetime = field(default_factory=datetime.utcnow)
    """Last updated timestamp in UTC."""

    def touch(self) -> None:
        """Update the `updated_at` timestamp."""
        self.updated_at = datetime.utcnow()

    def set_status(self, status: AgentStatus) -> None:
        """
        Set status and refresh timestamp.

        Args:
            status: New lifecycle status.
        """
        self.status = status
        self.touch()

    def add_log(self, entry: AgentLogEntry) -> None:
        """
        Append a new log entry and refresh timestamp.

        Args:
            entry: Log entry to append.
        """
        self.log_entries.append(entry)
        self.touch()

    def add_artifact(self, artifact_path: str) -> None:
        """
        Track a produced artifact path.

        Args:
            artifact_path: Filesystem path to track.
        """
        normalized = str(artifact_path).strip()
        if not normalized:
            return
        if normalized not in self.artifacts:
            self.artifacts.append(normalized)
            self.touch()


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """

    name: str = "base"
    description: str = "Base agent"

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        self.name = name or self.name
        self.description = description or self.description
        self.state = AgentState()
        self._stop_requested = False

    @abstractmethod
    def run(self, initial_context: AgentContext) -> AgentState:
        """
        Run the agent using an initial context.

        Args:
            initial_context: Initial runtime context.

        Returns:
            Final agent state for this run.
        """

    def pause(self) -> None:
        """Pause an active run."""
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        """Resume a paused run."""
        self.state.set_status(AgentStatus.RUNNING)

    def stop(self) -> None:
        """Request stop for an active run."""
        self._stop_requested = True
        self.state.set_status(AgentStatus.COMPLETED)
