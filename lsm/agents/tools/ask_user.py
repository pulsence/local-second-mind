"""
Tool for requesting clarification input from the user during agent execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict
from uuid import uuid4

from lsm.agents.interaction import InteractionRequest

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.agents.harness import AgentHarness


class AskUserTool(BaseTool):
    """Request clarification from the user and return the reply text."""

    name = "ask_user"
    description = "Ask the user for clarification and return the user response."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Question to show to the user.",
            },
            "context": {
                "type": "string",
                "description": "Optional rationale for why clarification is needed.",
            },
        },
        "required": ["prompt"],
    }

    def __init__(self) -> None:
        self._harness: AgentHarness | None = None

    def bind_harness(self, harness: AgentHarness) -> None:
        """Bind the active harness runtime for tool execution."""
        self._harness = harness

    def execute(self, args: Dict[str, Any]) -> str:
        harness = self._require_harness()
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")
        context = str(args.get("context", "")).strip()

        request = InteractionRequest(
            request_id=f"clarify-{uuid4().hex}",
            request_type="clarification",
            reason=context or "Agent requested clarification.",
            args_summary=json.dumps(
                {
                    "prompt": prompt,
                    "context": context,
                },
                ensure_ascii=True,
            ),
            prompt=prompt,
        )
        response = harness.request_interaction(request)
        if response.decision == "deny":
            raise PermissionError(
                str(response.user_message or "").strip()
                or "Clarification request was denied by the user."
            )
        return str(response.user_message or "")

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("ask_user tool is not bound to an active harness")
        return self._harness
