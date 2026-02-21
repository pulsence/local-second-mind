"""
Base classes and state models for agents.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .log_formatter import save_agent_log
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
    tier: str = "normal"
    tool_allowlist: Optional[set[str]] = None
    _always_available_tools: set[str] = {"ask_user"}

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None) -> None:
        self.name = name or self.name
        self.description = description or self.description
        self.tier = str(self.tier or "normal").strip().lower() or "normal"
        self.tool_allowlist = (
            {str(item).strip() for item in self.tool_allowlist if str(item).strip()}
            if self.tool_allowlist
            else None
        )
        self.state = AgentState()
        self._stop_requested = False
        self._stop_logged = False
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

    def _is_stop_requested(self) -> bool:
        """Return whether a stop was requested for this run."""
        return bool(self._stop_requested)

    def _handle_stop_request(self, message: str = "Stop requested; finishing current action.") -> bool:
        """
        Mark stop-request handling and ensure completed state.

        Returns:
            True when stop has been requested.
        """
        if not self._stop_requested:
            return False
        if not self._stop_logged:
            self._log(message)
            self._stop_logged = True
        self.state.set_status(AgentStatus.COMPLETED)
        return True

    def _log(
        self,
        content: str,
        actor: str = "agent",
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
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
                prompt=prompt,
                raw_response=raw_response,
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
        if self._stop_requested:
            return True
        if self.max_tokens_budget is None:
            return False
        return self._tokens_used >= int(self.max_tokens_budget)

    def _log_verbosity(self) -> str:
        """Return configured log verbosity from agent overrides."""
        overrides = getattr(self, "agent_overrides", None)
        if isinstance(overrides, dict):
            return overrides.get("log_verbosity", "normal")
        return "normal"

    def _llm_tier(self) -> str:
        """Return configured LLM tier from agent overrides or class default."""
        overrides = getattr(self, "agent_overrides", None)
        if isinstance(overrides, dict):
            tier = overrides.get("llm_tier") or overrides.get("tier")
            if tier is not None and str(tier).strip():
                return str(tier).strip().lower()
        return self.tier

    def _get_llm_selection(self) -> Dict[str, Any]:
        """Return LLM selection overrides for this agent."""
        overrides = getattr(self, "agent_overrides", None)
        if not isinstance(overrides, dict):
            overrides = {}
        return {
            "service": overrides.get("llm_service"),
            "tier": self._llm_tier(),
            "provider": overrides.get("llm_provider"),
            "model": overrides.get("llm_model"),
            "temperature": overrides.get("llm_temperature"),
            "max_tokens": overrides.get("llm_max_tokens"),
        }

    def _resolve_llm_config(self, llm_registry: Any) -> Any:
        """
        Resolve the effective LLM config for this agent.
        """
        selection = self._get_llm_selection()
        provider = selection.get("provider")
        model = selection.get("model")
        service = selection.get("service")
        tier = selection.get("tier")
        temperature = selection.get("temperature")
        max_tokens = selection.get("max_tokens")

        if provider and model:
            return llm_registry.resolve_direct(
                provider,
                model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if service:
            return llm_registry.resolve_service(str(service))
        if tier:
            try:
                return llm_registry.resolve_tier(str(tier))
            except Exception:
                warnings.warn(
                    f"LLM tier '{tier}' is not configured; falling back to 'default' service.",
                    stacklevel=2,
                )
                return llm_registry.resolve_service("default")
        return llm_registry.resolve_service("default")

    def _save_log(self) -> Path:
        """Persist the agent log to the default agent workspace path."""
        agent_config = getattr(self, "agent_config", None)
        if agent_config is None:
            raise RuntimeError("agent_config is required to save logs")
        log_path = save_agent_log(
            self.state.log_entries,
            agent_name=self.name,
            agents_folder=agent_config.agents_folder,
            verbosity=self._log_verbosity(),
        )
        self.state.add_artifact(str(log_path))
        return log_path


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
