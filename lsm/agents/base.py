"""
Base classes and state models for agents.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .models import AgentContext, AgentLogEntry

if TYPE_CHECKING:
    from .tools.base import ToolRegistry


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
    tool_allowlist: Optional[set[str]] = None
    _always_available_tools: set[str] = {"ask_user"}

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        self.name = name or self.name
        self.description = description or self.description
        self.tool_allowlist = (
            {str(item).strip() for item in self.tool_allowlist if str(item).strip()}
            if self.tool_allowlist
            else None
        )
        self.state = AgentState()
        self._stop_requested = False
        self._tokens_used = 0
        self.max_tokens_budget: Optional[int] = None

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

    def _log(
        self,
        content: str,
        actor: str = "agent",
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        action: Optional[str] = None,
        action_arguments: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a structured log entry."""
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor=actor,
                provider_name=provider_name,
                model_name=model_name,
                content=str(content),
                action=action,
                action_arguments=action_arguments,
            )
        )

    @staticmethod
    def _parse_json(value: str) -> Any:
        """Safely parse JSON text, returning None on errors."""
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def _consume_tokens(self, text: str) -> int:
        """Track approximate token usage using a 4-char heuristic."""
        estimated = max(1, len(str(text)) // 4)
        self._tokens_used += estimated
        return estimated

    def _budget_exhausted(self) -> bool:
        """Return True when the current token budget has been exhausted."""
        if self.max_tokens_budget is None:
            return False
        return self._tokens_used >= int(self.max_tokens_budget)

    def _get_tool_definitions(
        self,
        tool_registry: "ToolRegistry",
        tool_allowlist: Optional[set[str]] = None,
    ) -> list[Dict[str, Any]]:
        """
        Return tool definitions, optionally filtered by allowlist.
        """
        definitions = tool_registry.list_definitions()
        if not tool_allowlist:
            return definitions
        allowed = {name.strip() for name in tool_allowlist if str(name).strip()}
        allowed |= set(self._always_available_tools)
        return [item for item in definitions if str(item.get("name", "")).strip() in allowed]

    def _format_tool_definitions_for_prompt(
        self,
        tool_registry: "ToolRegistry",
        tool_allowlist: Optional[set[str]] = None,
    ) -> str:
        """
        Build a compact JSON block with tool names, descriptions, and argument schemas.
        """
        compact = []
        for item in self._get_tool_definitions(tool_registry, tool_allowlist):
            compact.append(
                {
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "input_schema": item.get("input_schema"),
                }
            )
        return json.dumps(compact, indent=2)

    def _parse_tool_selection(
        self,
        response: str,
        available_tool_names: Sequence[str],
    ) -> list[str]:
        """
        Parse tool-selection output of shape: {"tools": ["name", ...]}.
        """
        parsed = self._parse_json(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tools"), list):
            return []
        available_lookup = {name.lower(): name for name in available_tool_names}
        selected: list[str] = []
        for item in parsed["tools"]:
            key = str(item).strip().lower()
            if not key:
                continue
            normalized = available_lookup.get(key)
            if normalized and normalized not in selected:
                selected.append(normalized)
        return selected
