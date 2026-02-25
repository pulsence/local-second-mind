"""
Tool for awaiting completion of spawned sub-agents.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List

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
            "agent_ids": {
                "type": "array",
                "description": "Optional list of sub-agent run IDs to await.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Optional wait timeout in seconds.",
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

        agent_ids = args.get("agent_ids")
        if agent_ids is not None:
            if not isinstance(agent_ids, list):
                raise ValueError("agent_ids must be a list")
            normalized_ids = [str(item).strip() for item in agent_ids if str(item).strip()]
            if not normalized_ids:
                raise ValueError("agent_ids must contain at least one id")
            max_workers = max(1, int(getattr(harness.agent_config, "max_concurrent", 1)))
            results: List[dict] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        harness.await_sub_agent,
                        agent_id,
                        timeout_seconds=timeout_value,
                    ): agent_id
                    for agent_id in normalized_ids
                }
                for future in as_completed(future_map):
                    results.append(future.result())
            results.sort(key=lambda item: (item.get("agent_name", ""), item.get("agent_id", "")))
            return json.dumps({"runs": results}, indent=2, ensure_ascii=True)

        agent_id = str(args.get("agent_id", "")).strip()
        if not agent_id:
            raise ValueError("agent_id is required")

        payload = harness.await_sub_agent(agent_id, timeout_seconds=timeout_value)
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("await_agent tool is not bound to an active harness")
        return self._harness
