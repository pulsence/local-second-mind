"""
Dataclasses for agent runtime models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AgentLogEntry:
    """
    Structured entry for agent activity logs.
    """

    timestamp: datetime
    """UTC timestamp for this log event."""

    actor: str
    """Actor name for the event (e.g., 'agent', 'tool', 'llm', 'user')."""

    provider_name: Optional[str] = None
    """Optional LLM provider name for model events."""

    model_name: Optional[str] = None
    """Optional model name for model events."""

    content: str = ""
    """Human-readable event content."""

    prompt: Optional[str] = None
    """Optional LLM prompt text for debug logging."""

    raw_response: Optional[str] = None
    """Optional raw LLM response text for debug logging."""

    action: Optional[str] = None
    """Optional requested action (tool name, DONE, etc.)."""

    action_arguments: Optional[Dict[str, Any]] = None
    """Optional parsed action arguments payload."""


@dataclass
class ToolResponse:
    """
    Parsed response payload from an agent LLM/tool turn.
    """

    response: str
    """Natural-language response text from the model."""

    action: Optional[str] = None
    """Requested action (tool name or terminal marker)."""

    action_arguments: Dict[str, Any] = field(default_factory=dict)
    """JSON-parsed action arguments."""


@dataclass
class AgentContext:
    """
    Mutable context passed through an agent run.
    """

    messages: List[Dict[str, Any]] = field(default_factory=list)
    """Conversation/message history for the active run."""

    tool_definitions: List[Dict[str, Any]] = field(default_factory=list)
    """Tool schema metadata exposed to the agent LLM."""

    budget_tracking: Dict[str, Any] = field(default_factory=dict)
    """Token/cost counters and budget status for the run."""

    run_workspace: Optional[str] = None
    """Optional per-run workspace path for artifacts and agent outputs."""
