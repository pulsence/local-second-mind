"""
Tool for spawning sub-agent runs from a meta-agent harness.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.agents.harness import AgentHarness


class SpawnAgentTool(BaseTool):
    """Spawn a sub-agent execution with constrained permissions."""

    name = "spawn_agent"
    description = "Start a sub-agent run with params under the current harness."
    requires_permission = True
    risk_level = "exec"
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Registered sub-agent name (research|writing|synthesis|curator|meta).",
            },
            "params": {
                "type": "object",
                "description": "Optional parameter object passed to the sub-agent run.",
            },
            "agents": {
                "type": "array",
                "description": "Optional batch of sub-agent spawn requests.",
            },
        },
    }

    def __init__(self) -> None:
        self._harness: AgentHarness | None = None

    def bind_harness(self, harness: AgentHarness) -> None:
        """Bind the active harness runtime for tool execution."""
        self._harness = harness

    def execute(self, args: Dict[str, Any]) -> str:
        harness = self._require_harness()
        batch = args.get("agents")
        if batch is not None:
            if not isinstance(batch, list):
                raise ValueError("agents must be a list")
            results = []
            for entry in batch:
                if not isinstance(entry, dict):
                    raise ValueError("agents entries must be objects")
                agent_name = str(entry.get("name", "")).strip()
                if not agent_name:
                    raise ValueError("agents entry name is required")
                raw_params = entry.get("params", {})
                if raw_params is None:
                    raw_params = {}
                if not isinstance(raw_params, dict):
                    raise ValueError("agents entry params must be an object")
                results.append(harness.spawn_sub_agent(agent_name, dict(raw_params)))
            return json.dumps({"runs": results}, indent=2, ensure_ascii=True)

        agent_name = str(args.get("name", "")).strip()
        if not agent_name:
            raise ValueError("name is required")

        raw_params = args.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError("params must be an object")

        payload = harness.spawn_sub_agent(agent_name, dict(raw_params))
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("spawn_agent tool is not bound to an active harness")
        return self._harness
