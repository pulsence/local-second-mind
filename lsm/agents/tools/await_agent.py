"""
Tool for awaiting completion of spawned sub-agents.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.agents.harness import AgentHarness


class AwaitAgentTool(BaseTool):
    """Wait for a spawned sub-agent to complete."""

    name = "await_agent"
    description = "Block until a spawned sub-agent run completes."
    requires_permission = True
    risk_level = "exec"
    input_schema = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Sub-agent run ID returned by spawn_agent.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional wait timeout in seconds.",
            },
        },
        "required": ["agent_id"],
    }

    def __init__(self) -> None:
        self._harness: AgentHarness | None = None

    def bind_harness(self, harness: AgentHarness) -> None:
        """Bind the active harness runtime for tool execution."""
        self._harness = harness

    def execute(self, args: Dict[str, Any]) -> str:
        harness = self._require_harness()
        agent_id = str(args.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent_id is required")

        timeout_seconds = args.get("timeout_seconds")
        if timeout_seconds is not None:
            try:
                timeout_value = float(timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError("timeout_seconds must be numeric") from exc
            if timeout_value <= 0:
                raise ValueError("timeout_seconds must be > 0")
        else:
            timeout_value = None

        payload = harness.await_sub_agent(agent_id, timeout_seconds=timeout_value)
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("await_agent tool is not bound to an active harness")
        return self._harness
