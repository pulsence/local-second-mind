"""
Agent runtime harness for executing tool-using agent loops.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.config.models import AgentConfig, LLMRegistryConfig
from lsm.logging import get_logger
from lsm.providers.factory import create_provider

from .base import AgentState, AgentStatus
from .models import AgentContext, AgentLogEntry, ToolResponse
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox

logger = get_logger(__name__)


class AgentHarness:
    """
    Runtime engine for agent execution loops.
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        llm_registry: LLMRegistryConfig,
        sandbox: ToolSandbox,
        agent_name: str = "agent",
    ) -> None:
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.llm_registry = llm_registry
        self.sandbox = sandbox
        self.agent_name = agent_name

        self.state = AgentState()
        self.context: Optional[AgentContext] = None
        self._state_path: Optional[Path] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def run(self, initial_context: AgentContext) -> AgentState:
        """
        Run the agent loop synchronously.

        Args:
            initial_context: Initial context for the run.

        Returns:
            Final agent state.
        """
        with self._lock:
            self._stop_event.clear()
            self.context = initial_context
            self.context.tool_definitions = self.tool_registry.list_definitions()
            if not self.context.budget_tracking:
                self.context.budget_tracking = {}
            self.context.budget_tracking.setdefault(
                "max_tokens_budget",
                self.agent_config.max_tokens_budget,
            )
            self.context.budget_tracking.setdefault("tokens_used", 0)
            self.context.budget_tracking.setdefault("iterations", 0)
            self.state.set_status(AgentStatus.RUNNING)
            self._ensure_state_path()
            self.save_state()

        try:
            llm_config = self.llm_registry.resolve_service("default")
            llm_provider = create_provider(llm_config)
            for _ in range(self.agent_config.max_iterations):
                if self._stop_event.is_set():
                    self.state.set_status(AgentStatus.COMPLETED)
                    break

                while self.state.status == AgentStatus.PAUSED and not self._stop_event.is_set():
                    time.sleep(0.05)

                if self._stop_event.is_set():
                    self.state.set_status(AgentStatus.COMPLETED)
                    break

                self.context.budget_tracking["iterations"] += 1
                messages_for_llm = self._prepare_messages(self.context)
                raw_response = llm_provider.synthesize(
                    question=self._messages_to_prompt(messages_for_llm),
                    context=self._tools_context_block(),
                    mode="insight",
                )
                self._consume_tokens(raw_response)
                tool_response = self._parse_tool_response(raw_response)
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="llm",
                        provider_name=llm_provider.name,
                        model_name=llm_provider.model,
                        content=tool_response.response,
                        action=tool_response.action,
                        action_arguments=tool_response.action_arguments,
                    )
                )
                self.context.messages.append(
                    {"role": "assistant", "content": tool_response.response}
                )

                action = (tool_response.action or "").strip()
                if not action or action.upper() == "DONE":
                    self.state.set_status(AgentStatus.COMPLETED)
                    self.save_state()
                    return self.state

                tool = self.tool_registry.lookup(action)
                tool_output = self.sandbox.execute(tool, tool_response.action_arguments)
                self._consume_tokens(tool_output)
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="tool",
                        content=tool_output,
                        action=action,
                        action_arguments=tool_response.action_arguments,
                    )
                )
                self.context.messages.append(
                    {
                        "role": "tool",
                        "name": action,
                        "content": tool_output,
                    }
                )

                if self._budget_exhausted():
                    self.state.set_status(AgentStatus.COMPLETED)
                    self._append_log(
                        AgentLogEntry(
                            timestamp=datetime.utcnow(),
                            actor="agent",
                            content="Stopping due to token budget exhaustion.",
                        )
                    )
                    break

                self.save_state()

            if self.state.status not in {AgentStatus.COMPLETED, AgentStatus.FAILED}:
                self.state.set_status(AgentStatus.COMPLETED)
            self.save_state()
            return self.state
        except Exception as exc:
            logger.error(f"Agent harness failed: {exc}")
            self.state.set_status(AgentStatus.FAILED)
            self._append_log(
                AgentLogEntry(
                    timestamp=datetime.utcnow(),
                    actor="agent",
                    content=f"Execution failed: {exc}",
                )
            )
            self.save_state()
            return self.state

    def start_background(self, initial_context: AgentContext) -> threading.Thread:
        """
        Run the agent loop in a background thread.

        Args:
            initial_context: Initial context for execution.

        Returns:
            Started thread handle.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Agent harness is already running")
        self._thread = threading.Thread(
            target=self.run,
            args=(initial_context,),
            daemon=True,
            name=f"AgentHarness-{self.agent_name}",
        )
        self._thread.start()
        return self._thread

    def pause(self) -> None:
        """Pause execution and persist state."""
        self.state.set_status(AgentStatus.PAUSED)
        self.save_state()

    def resume(self) -> None:
        """Resume execution and persist state."""
        self.state.set_status(AgentStatus.RUNNING)
        self.save_state()

    def stop(self) -> None:
        """Stop execution and persist state."""
        self._stop_event.set()
        self.state.set_status(AgentStatus.COMPLETED)
        self.save_state()

    def get_state_path(self) -> Optional[Path]:
        """Return the current state file path."""
        return self._state_path

    def save_state(self) -> Optional[Path]:
        """
        Persist state to a JSON file.

        Returns:
            Path written, or None if state path is not initialized.
        """
        if self._state_path is None:
            return None
        payload = {
            "agent_name": self.agent_name,
            "status": self.state.status.value,
            "current_task": self.state.current_task,
            "created_at": self.state.created_at.isoformat(),
            "updated_at": self.state.updated_at.isoformat(),
            "context": {
                "messages": self.context.messages if self.context else [],
                "tool_definitions": self.context.tool_definitions if self.context else [],
                "budget_tracking": self.context.budget_tracking if self.context else {},
            },
            "log_entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "provider_name": entry.provider_name,
                    "model_name": entry.model_name,
                    "content": entry.content,
                    "action": entry.action,
                    "action_arguments": entry.action_arguments,
                }
                for entry in self.state.log_entries
            ],
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self._state_path

    def _ensure_state_path(self) -> None:
        if self._state_path is not None:
            return
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.agent_name}_{timestamp}_state.json"
        self._state_path = self.agent_config.agents_folder / filename

    def _append_log(self, entry: AgentLogEntry) -> None:
        self.state.add_log(entry)

    def _parse_tool_response(self, raw_response: str) -> ToolResponse:
        text = str(raw_response or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ToolResponse(response=text, action=None, action_arguments={})

        if not isinstance(parsed, dict):
            return ToolResponse(response=text, action=None, action_arguments={})

        action_arguments = parsed.get("action_arguments", {})
        if not isinstance(action_arguments, dict):
            action_arguments = {}
        return ToolResponse(
            response=str(parsed.get("response", "")),
            action=(
                str(parsed.get("action")).strip()
                if parsed.get("action") is not None
                else None
            ),
            action_arguments=action_arguments,
        )

    def _messages_to_prompt(self, messages: list[Dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}")
        lines.append(
            "Respond with strict JSON: "
            '{"response":"...","action":"TOOL_OR_DONE","action_arguments":{}}'
        )
        return "\n".join(lines)

    def _prepare_messages(self, context: AgentContext) -> list[Dict[str, Any]]:
        messages = list(context.messages)
        if self.agent_config.context_window_strategy == "fresh":
            if len(messages) <= 6:
                return messages
            return messages[-6:]

        if len(messages) <= 12:
            return messages

        older = messages[:-8]
        recent = messages[-8:]
        summary = f"Compacted {len(older)} earlier messages."
        return [{"role": "system", "content": summary}] + recent

    def _tools_context_block(self) -> str:
        return json.dumps(self.tool_registry.list_definitions(), indent=2)

    def _consume_tokens(self, text: str) -> None:
        if self.context is None:
            return
        estimated = max(1, len(str(text)) // 4)
        self.context.budget_tracking["tokens_used"] = (
            int(self.context.budget_tracking.get("tokens_used", 0)) + estimated
        )

    def _budget_exhausted(self) -> bool:
        if self.context is None:
            return False
        used = int(self.context.budget_tracking.get("tokens_used", 0))
        limit = int(self.context.budget_tracking.get("max_tokens_budget", self.agent_config.max_tokens_budget))
        return used >= limit

