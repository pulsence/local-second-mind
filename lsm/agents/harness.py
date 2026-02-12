"""
Agent runtime harness for executing tool-using agent loops.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from lsm.config.models import AgentConfig, LLMRegistryConfig, VectorDBConfig
from lsm.logging import get_logger
from lsm.providers.factory import create_provider

from .log_redactor import redact_secrets
from .base import AgentState, AgentStatus
from .memory import BaseMemoryStore, MemoryContextBuilder, create_memory_store
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
        tool_allowlist: Optional[Set[str]] = None,
        vectordb_config: Optional[VectorDBConfig] = None,
        memory_store: Optional[BaseMemoryStore] = None,
        memory_context_builder: Optional[MemoryContextBuilder] = None,
    ) -> None:
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.llm_registry = llm_registry
        self.sandbox = sandbox
        self.agent_name = agent_name
        self.tool_allowlist = self._normalize_allowlist(tool_allowlist)
        self.vectordb_config = vectordb_config

        self.state = AgentState()
        self.context: Optional[AgentContext] = None
        self.memory_store: Optional[BaseMemoryStore] = memory_store
        self.memory_context_builder: Optional[MemoryContextBuilder] = None
        self._state_path: Optional[Path] = None
        self._run_root: Optional[Path] = None
        self._workspace_path: Optional[Path] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._tool_usage_counts: Dict[str, int] = {}
        self._tool_sequence: list[str] = []
        self._permission_events: list[Dict[str, Any]] = []

        self._initialize_memory_context_builder(memory_context_builder)

    def run(self, initial_context: AgentContext) -> AgentState:
        """
        Run the agent loop synchronously.

        Args:
            initial_context: Initial context for the run.

        Returns:
            Final agent state.
        """
        started_at = datetime.utcnow()
        with self._lock:
            self._stop_event.clear()
            self.context = initial_context
            self._tool_usage_counts = {}
            self._tool_sequence = []
            self._permission_events = []
            self.context.tool_definitions = self._list_tool_definitions()
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
            if self._workspace_path is not None:
                self.context.run_workspace = str(self._workspace_path)
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
                standing_context_block = self._build_standing_context_block()
                messages_for_llm = self._prepare_messages(self.context)
                raw_response = llm_provider.synthesize(
                    question=self._messages_to_prompt(
                        messages_for_llm,
                        standing_context_block=standing_context_block,
                    ),
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
                    break

                if not self._is_tool_allowed(action):
                    denial_reason = f"Tool '{action}' is not allowed for this harness run"
                    self._record_permission_decision(action, allowed=False, reason=denial_reason)
                    raise PermissionError(denial_reason)

                tool = self.tool_registry.lookup(action)
                try:
                    tool_output = self.sandbox.execute(tool, tool_response.action_arguments)
                except PermissionError as exc:
                    self._record_permission_decision(
                        action,
                        allowed=False,
                        reason=str(exc),
                    )
                    raise

                self._record_permission_decision(
                    action,
                    allowed=True,
                    reason="Tool execution allowed",
                )
                self._record_tool_execution(action)
                self._track_artifacts_from_sandbox()
                self._consume_tokens(tool_output)
                redacted_tool_output = redact_secrets(tool_output)
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="tool",
                        content=redacted_tool_output,
                        action=action,
                        action_arguments=tool_response.action_arguments,
                    )
                )
                self.context.messages.append(
                    {
                        "role": "tool",
                        "name": action,
                        "content": redacted_tool_output,
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
        finally:
            self._write_run_summary(started_at)

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

    def get_workspace_path(self) -> Optional[Path]:
        """Return the per-run workspace path."""
        return self._workspace_path

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
                "messages": self._serialize_context_messages() if self.context else [],
                "tool_definitions": self.context.tool_definitions if self.context else [],
                "budget_tracking": self.context.budget_tracking if self.context else {},
                "run_workspace": self.context.run_workspace if self.context else None,
            },
            "artifacts": list(self.state.artifacts),
            "log_entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "provider_name": entry.provider_name,
                    "model_name": entry.model_name,
                    "content": redact_secrets(entry.content),
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
        run_name = f"{self.agent_name}_{timestamp}"
        self._run_root = self.agent_config.agents_folder / run_name
        self._workspace_path = self._run_root / "workspace"
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        filename = f"{run_name}_state.json"
        self._state_path = self._run_root / filename
        self.state.add_artifact(str(self._workspace_path))

    def _append_log(self, entry: AgentLogEntry) -> None:
        entry.content = redact_secrets(entry.content)
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

    def _messages_to_prompt(
        self,
        messages: list[Dict[str, Any]],
        *,
        standing_context_block: str = "",
    ) -> str:
        lines = []
        if standing_context_block:
            lines.append(
                "SYSTEM: Use the following standing context from persistent agent memory."
            )
            lines.append(standing_context_block)
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
        if self.context is not None:
            return json.dumps(self.context.tool_definitions, indent=2)
        return json.dumps(self._list_tool_definitions(), indent=2)

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

    def _serialize_context_messages(self) -> list[Dict[str, Any]]:
        if self.context is None:
            return []
        sanitized: list[Dict[str, Any]] = []
        for message in self.context.messages:
            serialized = dict(message)
            if "content" in serialized:
                serialized["content"] = redact_secrets(str(serialized.get("content", "")))
            sanitized.append(serialized)
        return sanitized

    def _track_artifacts_from_sandbox(self) -> None:
        result = self.sandbox.last_execution_result
        if result is None:
            return
        for artifact in result.artifacts:
            self.state.add_artifact(str(artifact))

    def _record_tool_execution(self, tool_name: str) -> None:
        normalized = str(tool_name).strip()
        if not normalized:
            return
        self._tool_usage_counts[normalized] = self._tool_usage_counts.get(normalized, 0) + 1
        self._tool_sequence.append(normalized)

    def _record_permission_decision(self, tool_name: str, *, allowed: bool, reason: str) -> None:
        normalized = str(tool_name).strip()
        if not normalized:
            return
        self._permission_events.append(
            {
                "tool_name": normalized,
                "allowed": bool(allowed),
                "reason": str(reason),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def _write_run_summary(self, started_at: datetime) -> Optional[Path]:
        if self._run_root is None:
            return None
        finished_at = datetime.utcnow()
        duration_seconds = max(0.0, (finished_at - started_at).total_seconds())
        summary_path = self._run_root / "run_summary.json"

        approval_count = 0
        denial_count = 0
        by_tool: Dict[str, Dict[str, int]] = {}
        for event in self._permission_events:
            tool_name = str(event.get("tool_name", "")).strip()
            if not tool_name:
                continue
            entry = by_tool.setdefault(tool_name, {"approvals": 0, "denials": 0})
            if bool(event.get("allowed", False)):
                approval_count += 1
                entry["approvals"] += 1
            else:
                denial_count += 1
                entry["denials"] += 1

        budget = self.context.budget_tracking if self.context is not None else {}
        summary = {
            "agent_name": self.agent_name,
            "topic": self._extract_topic(self.context) if self.context is not None else "",
            "status": self.state.status.value,
            "run_outcome": (
                "failed" if self.state.status == AgentStatus.FAILED else "completed"
            ),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": round(duration_seconds, 3),
            "tools_used": dict(self._tool_usage_counts),
            "tool_sequence": list(self._tool_sequence),
            "approvals_denials": {
                "approvals": approval_count,
                "denials": denial_count,
                "by_tool": by_tool,
                "events": self._permission_events,
            },
            "artifacts_created": list(self.state.artifacts),
            "token_usage": {
                "tokens_used": int(budget.get("tokens_used", 0)),
                "max_tokens_budget": int(
                    budget.get("max_tokens_budget", self.agent_config.max_tokens_budget)
                ),
                "iterations": int(budget.get("iterations", 0)),
            },
            "constraints": self._extract_constraints(
                self._extract_topic(self.context) if self.context is not None else ""
            ),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.state.add_artifact(str(summary_path))
        return summary_path

    @staticmethod
    def _normalize_allowlist(tool_allowlist: Optional[Set[str]]) -> Optional[Set[str]]:
        if not tool_allowlist:
            return None
        normalized = {str(name).strip() for name in tool_allowlist if str(name).strip()}
        return normalized or None

    def _list_tool_definitions(self) -> list[Dict[str, Any]]:
        definitions = self.tool_registry.list_definitions()
        if self.tool_allowlist is None:
            return definitions
        return [
            definition
            for definition in definitions
            if str(definition.get("name", "")).strip() in self.tool_allowlist
        ]

    def _is_tool_allowed(self, tool_name: str) -> bool:
        if self.tool_allowlist is None:
            return True
        return str(tool_name).strip() in self.tool_allowlist

    def _initialize_memory_context_builder(
        self,
        memory_context_builder: Optional[MemoryContextBuilder],
    ) -> None:
        if not self.agent_config.memory.enabled:
            self.memory_context_builder = None
            return

        if memory_context_builder is not None:
            self.memory_context_builder = memory_context_builder
            if self.memory_store is None:
                self.memory_store = memory_context_builder.store
            return

        if self.memory_store is None and self.vectordb_config is not None:
            try:
                self.memory_store = create_memory_store(
                    self.agent_config,
                    self.vectordb_config,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize memory store for harness '%s': %s",
                    self.agent_name,
                    exc,
                )
                self.memory_store = None

        if self.memory_store is not None:
            self.memory_context_builder = MemoryContextBuilder(self.memory_store)

    def _build_standing_context_block(self) -> str:
        if self.context is None or self.memory_context_builder is None:
            return ""
        topic = self._extract_topic(self.context)
        memory_token_budget = max(
            200,
            min(2000, int(self.agent_config.max_tokens_budget // 8)),
        )
        try:
            return self.memory_context_builder.build(
                agent_name=self.agent_name,
                topic=topic,
                limit=8,
                token_budget=memory_token_budget,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build memory context block for harness '%s': %s",
                self.agent_name,
                exc,
            )
            return ""

    @staticmethod
    def _extract_topic(context: AgentContext) -> str:
        for message in reversed(context.messages):
            if str(message.get("role", "")).strip().lower() != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content:
                return content
        return ""

    @staticmethod
    def _extract_constraints(topic: str) -> list[str]:
        text = str(topic or "").strip()
        if not text:
            return []
        lowered = text.lower()
        markers = [
            "avoid ",
            "must ",
            "should ",
            "do not ",
            "don't ",
            "without ",
            "no ",
        ]
        constraints: list[str] = []
        for marker in markers:
            index = lowered.find(marker)
            if index < 0:
                continue
            segment = text[index:].strip()
            if segment:
                constraints.append(segment.rstrip("."))
        seen: set[str] = set()
        deduped: list[str] = []
        for item in constraints:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:8]
