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
            "agent_ids": {
                "type": "array",
                "description": "Optional list of sub-agent run IDs to collect.",
            },
            "pattern": {
                "type": "string",
                "description": "Optional glob pattern for filtering artifact paths.",
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
        pattern = str(args.get("pattern", "*") or "*").strip() or "*"

        agent_ids = args.get("agent_ids")
        if agent_ids is not None:
            if not isinstance(agent_ids, list):
                raise ValueError("agent_ids must be a list")
            normalized_ids = [str(item).strip() for item in agent_ids if str(item).strip()]
            if not normalized_ids:
                raise ValueError("agent_ids must contain at least one id")
            runs = []
            for agent_id in normalized_ids:
                artifacts = harness.collect_sub_agent_artifacts(agent_id, pattern=pattern)
                runs.append(
                    {
                        "agent_id": agent_id,
                        "pattern": pattern,
                        "artifacts": artifacts,
                    }
                )
            runs.sort(key=lambda item: item.get("agent_id", ""))
            return json.dumps({"runs": runs}, indent=2, ensure_ascii=True)

        agent_id = str(args.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent_id is required")

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
