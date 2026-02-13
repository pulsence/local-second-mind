"""
Tool for collecting artifacts produced by spawned sub-agents.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.agents.harness import AgentHarness


class CollectArtifactsTool(BaseTool):
    """Collect artifact paths from a spawned sub-agent run."""

    name = "collect_artifacts"
    description = "Collect artifacts from a spawned sub-agent by run ID and optional glob pattern."
    requires_permission = True
    risk_level = "exec"
    input_schema = {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "Sub-agent run ID returned by spawn_agent.",
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern for filtering artifact paths.",
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
        pattern = str(args.get("pattern", "*") or "*").strip() or "*"

        artifacts = harness.collect_sub_agent_artifacts(agent_id, pattern=pattern)
        payload = {
            "agent_id": agent_id,
            "pattern": pattern,
            "artifacts": artifacts,
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("collect_artifacts tool is not bound to an active harness")
        return self._harness
