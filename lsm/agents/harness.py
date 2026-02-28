"""
Agent runtime harness for executing tool-using agent loops.
"""

from __future__ import annotations

import fnmatch
import json
import threading
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

from lsm.config.models import AgentConfig, LLMRegistryConfig, LSMConfig, VectorDBConfig
from lsm.config.models.agents import SandboxConfig
from lsm.logging import get_logger
from lsm.providers.factory import create_provider
from lsm.utils.paths import resolve_path

from .interaction import InteractionChannel, InteractionRequest, InteractionResponse
from .log_formatter import save_agent_log
from .log_redactor import redact_secrets
from .base import AgentState, AgentStatus
from .factory import create_agent
from .memory import BaseMemoryStore, MemoryContextBuilder, create_memory_store
from .models import AgentContext, AgentLogEntry, ToolResponse
from .phase import PhaseResult
from .tools.base import ToolRegistry
from .tools.sandbox import ToolSandbox
from .workspace import ensure_agent_workspace

logger = get_logger(__name__)


@dataclass
class _SubAgentRun:
    """
    In-memory tracking record for a spawned sub-agent.
    """

    agent_id: str
    agent_name: str
    params: Dict[str, Any]
    harness: AgentHarness
    thread: threading.Thread
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AgentHarness:
    """
    Runtime engine for agent execution loops.
    """

    _ALWAYS_AVAILABLE_TOOLS = {"ask_user"}
    _BUILTIN_QUERY_TOOL_NAMES = frozenset(
        {"query_knowledge_base", "query_llm", "query_remote_chain"}
    )

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_registry: ToolRegistry,
        llm_registry: LLMRegistryConfig,
        sandbox: ToolSandbox,
        agent_name: str = "agent",
        tool_allowlist: Optional[Set[str]] = None,
        remote_source_allowlist: Optional[Set[str]] = None,
        lsm_config: Optional[LSMConfig] = None,
        llm_service: Optional[str] = None,
        llm_tier: Optional[str] = "normal",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        vectordb_config: Optional[VectorDBConfig] = None,
        memory_store: Optional[BaseMemoryStore] = None,
        memory_context_builder: Optional[MemoryContextBuilder] = None,
        interaction_channel: Optional[InteractionChannel] = None,
        log_callback: Optional[Callable[[AgentLogEntry], None]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.agent_config = agent_config
        self.tool_registry = tool_registry
        self.llm_registry = llm_registry
        self.sandbox = sandbox
        self.agent_name = agent_name
        self.tool_allowlist = self._normalize_allowlist(tool_allowlist)
        self.remote_source_allowlist = self._normalize_remote_sources(remote_source_allowlist)
        self.lsm_config = lsm_config
        self.llm_service = str(llm_service).strip() if llm_service else None
        self.llm_tier = str(llm_tier or "normal").strip().lower() if llm_tier else None
        self.llm_provider = str(llm_provider).strip() if llm_provider else None
        self.llm_model = str(llm_model).strip() if llm_model else None
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.vectordb_config = vectordb_config
        self.interaction_channel = interaction_channel or sandbox.interaction_channel
        self.log_callback = log_callback
        self.system_prompt = str(system_prompt).strip() if system_prompt else None

        self.state = AgentState()
        self.context: Optional[AgentContext] = None
        self.memory_store: Optional[BaseMemoryStore] = memory_store
        self.memory_context_builder: Optional[MemoryContextBuilder] = None
        self._state_path: Optional[Path] = None
        self._run_root: Optional[Path] = None
        self._workspace_path: Optional[Path] = None
        self._agent_workspace_root: Optional[Path] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._tool_usage_counts: Dict[str, int] = {}
        self._tool_sequence: list[str] = []
        self._permission_events: list[Dict[str, Any]] = []
        self._sub_agent_runs: Dict[str, _SubAgentRun] = {}
        self._sub_agent_counter = 0
        self._meta_tool_names = {"spawn_agent", "await_agent", "collect_artifacts"}
        self._context_histories: Dict[Optional[str], list] = {}
        self.sandbox.set_interaction_channel(
            self.interaction_channel,
            waiting_state_callback=self._set_waiting_for_user,
        )

        self._initialize_memory_context_builder(memory_context_builder)
        self._bind_runtime_tools()

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
            self.context = initial_context
            self._tool_usage_counts = {}
            self._tool_sequence = []
            self._permission_events = []
            self._context_histories = {None: list(initial_context.messages)}
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
            self._ensure_agent_workspace()
            self._ensure_state_path()
            if self._workspace_path is not None:
                self.context.run_workspace = str(self._workspace_path)
            self.save_state()

        try:
            while self.state.status == AgentStatus.PAUSED and not self._stop_event.is_set():
                time.sleep(0.05)

            if not self._stop_event.is_set():
                self.run_bounded(
                    user_message="",
                    tool_names=None,
                    max_iterations=max(1, int(self.agent_config.max_iterations)),
                    continue_context=True,
                    context_label=None,
                )
            else:
                self.state.set_status(AgentStatus.COMPLETED)

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
            self._write_agent_log()
            self._write_run_summary(started_at)

    def _ensure_context(self) -> AgentContext:
        if self.context is None:
            self.context = AgentContext(
                messages=[],
                tool_definitions=self._list_tool_definitions(),
                budget_tracking={},
            )
        if not self.context.tool_definitions:
            self.context.tool_definitions = self._list_tool_definitions()
        if not self.context.budget_tracking:
            self.context.budget_tracking = {}
        self.context.budget_tracking.setdefault(
            "max_tokens_budget",
            self.agent_config.max_tokens_budget,
        )
        self.context.budget_tracking.setdefault("tokens_used", 0)
        self.context.budget_tracking.setdefault("iterations", 0)
        return self.context

    def run_bounded(
        self,
        user_message: str = "",
        tool_names: Optional[list[str]] = None,
        max_iterations: int = 10,
        continue_context: bool = True,
        context_label: Optional[str] = None,
        direct_tool_calls: Optional[list[dict]] = None,
    ) -> PhaseResult:
        if self._stop_event.is_set():
            return PhaseResult(final_text="", tool_calls=[], stop_reason="stop_requested")

        if direct_tool_calls is not None:
            return self._run_bounded_tool_only(direct_tool_calls)

        return self._run_bounded_llm_mode(
            user_message=user_message,
            tool_names=tool_names,
            max_iterations=max_iterations,
            continue_context=continue_context,
            context_label=context_label,
        )

    def _run_bounded_tool_only(
        self,
        direct_tool_calls: list[dict],
    ) -> PhaseResult:
        tool_results: list[dict] = []
        for call in direct_tool_calls:
            tool_name = str(call.get("name", "")).strip()
            if not tool_name:
                tool_results.append({
                    "error": "Tool call missing 'name' field",
                })
                continue

            if not self._is_tool_allowed(tool_name):
                tool_results.append({
                    "name": tool_name,
                    "error": f"Tool '{tool_name}' is not allowed",
                })
                continue

            tool = self.tool_registry.lookup(tool_name)
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            try:
                output = self.sandbox.execute(tool, arguments)
                tool_results.append({
                    "name": tool_name,
                    "result": output,
                })
            except Exception as exc:
                tool_results.append({
                    "name": tool_name,
                    "error": str(exc),
                })

        return PhaseResult(final_text="", tool_calls=tool_results, stop_reason="done")

    def _run_bounded_llm_mode(
        self,
        user_message: str,
        tool_names: Optional[list[str]],
        max_iterations: int,
        continue_context: bool,
        context_label: Optional[str],
    ) -> PhaseResult:
        context = self._ensure_context()
        if not continue_context or context_label not in self._context_histories:
            self._context_histories[context_label] = []

        history = self._context_histories[context_label]

        if user_message:
            history.append({"role": "user", "content": user_message})
            context.messages.append({"role": "user", "content": user_message})

        llm_config = self._resolve_llm_config()
        llm_provider = create_provider(llm_config)
        tool_calls_accumulated: list[dict] = []
        final_text = ""
        stop_reason = "done"
        tool_names_restricted = set(tool_names) if tool_names is not None else None
        tool_definitions = self._list_tool_definitions(tool_names=tool_names_restricted)

        for iteration in range(max_iterations):
            context.budget_tracking["iterations"] = int(
                context.budget_tracking.get("iterations", 0)
            ) + 1
            if self._budget_exhausted():
                stop_reason = "budget_exhausted"
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="agent",
                        content="Stopping due to token budget exhaustion.",
                    )
                )
                break

            if self._stop_event.is_set():
                stop_reason = "stop_requested"
                break

            standing_context_block = self._build_standing_context_block()
            messages_for_llm = self._prepare_messages_from_history(history)
            raw_response, system_prompt, user_prompt = self._call_llm(
                llm_provider,
                messages_for_llm=messages_for_llm,
                standing_context_block=standing_context_block,
                tool_definitions=tool_definitions,
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
                    raw_response=str(raw_response),
                )
            )

            history.append({"role": "assistant", "content": tool_response.response})
            context.messages.append({"role": "assistant", "content": tool_response.response})

            action = (tool_response.action or "").strip()
            if not action or action.upper() == "DONE":
                final_text = tool_response.response
                break

            if not self._is_tool_allowed(action):
                denial_reason = f"Tool '{action}' is not allowed for this harness run"
                self._record_permission_decision(action, allowed=False, reason=denial_reason)
                tool_calls_accumulated.append({
                    "name": action,
                    "error": denial_reason,
                })
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="agent",
                        content=denial_reason,
                        action=action,
                    )
                )
                feedback = f"Error: {denial_reason}"
                history.append({"role": "tool", "name": action, "content": feedback})
                context.messages.append({"role": "tool", "name": action, "content": feedback})
                continue

            if tool_names_restricted is not None and action not in tool_names_restricted:
                denial_reason = f"Tool '{action}' is not in the requested tool_names list"
                self._record_permission_decision(action, allowed=False, reason=denial_reason)
                tool_calls_accumulated.append({
                    "name": action,
                    "error": denial_reason,
                })
                self._append_log(
                    AgentLogEntry(
                        timestamp=datetime.utcnow(),
                        actor="agent",
                        content=denial_reason,
                        action=action,
                    )
                )
                feedback = f"Error: {denial_reason}"
                history.append({"role": "tool", "name": action, "content": feedback})
                context.messages.append({"role": "tool", "name": action, "content": feedback})
                continue

            tool = self.tool_registry.lookup(action)
            try:
                tool_output = self.sandbox.execute(tool, tool_response.action_arguments)
            except PermissionError as exc:
                self._record_permission_decision(
                    action,
                    allowed=False,
                    reason=str(exc),
                )
                tool_calls_accumulated.append({
                    "name": action,
                    "error": str(exc),
                })
                if self._stop_event.is_set():
                    stop_reason = "stop_requested"
                    break
                stop_reason = "done"
                break

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

            tool_calls_accumulated.append({
                "name": action,
                "result": redacted_tool_output,
            })

            history.append({
                "role": "tool",
                "name": action,
                "content": redacted_tool_output,
            })
            context.messages.append({
                "role": "tool",
                "name": action,
                "content": redacted_tool_output,
            })

        else:
            stop_reason = "max_iterations"

        if history and history[-1].get("role") == "user":
            if not final_text:
                final_text = history[-1].get("content", "")

        return PhaseResult(
            final_text=final_text,
            tool_calls=tool_calls_accumulated,
            stop_reason=stop_reason,
        )

    def _prepare_messages_from_history(self, history: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        messages = list(history)
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
        if self.interaction_channel is not None:
            self.interaction_channel.cancel_pending("Agent stopped")
        for run in list(self._sub_agent_runs.values()):
            try:
                run.harness.stop()
            except Exception:
                logger.exception(
                    "Failed to stop spawned sub-agent '%s' (%s)",
                    run.agent_name,
                    run.agent_id,
                )
        self.state.set_status(AgentStatus.COMPLETED)
        self.save_state()

    def get_state_path(self) -> Optional[Path]:
        """Return the current state file path."""
        return self._state_path

    def get_workspace_path(self) -> Optional[Path]:
        """Return the per-run workspace path."""
        return self._workspace_path

    def spawn_sub_agent(self, agent_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a sub-agent run in a background thread.

        Args:
            agent_name: Registered sub-agent name.
            params: Optional run parameters.

        Returns:
            Spawn record payload with run identifier and state.
        """
        normalized_name = str(agent_name or "").strip().lower()
        if not normalized_name:
            raise ValueError("agent_name is required")
        normalized_params = dict(params or {})

        with self._lock:
            self._ensure_state_path()
            if self._run_root is None or self._workspace_path is None:
                raise RuntimeError("Harness run paths are not initialized")
            self._sub_agent_counter += 1
            agent_id = f"{normalized_name}_{self._sub_agent_counter:03d}"
            sub_root = self._run_root / "sub_agents" / agent_id

        sub_root.mkdir(parents=True, exist_ok=True)

        child_sandbox = self._build_sub_agent_sandbox(
            sub_root=sub_root,
            shared_workspace=self._workspace_path,
            params=normalized_params,
        )
        child_registry = self._build_sub_agent_tool_registry()
        child_agent_config = replace(self.agent_config, agents_folder=sub_root)
        child_agent = create_agent(
            name=normalized_name,
            llm_registry=self.llm_registry,
            tool_registry=child_registry,
            sandbox=child_sandbox,
            agent_config=child_agent_config,
            lsm_config=self.lsm_config,
        )
        child_selection: Dict[str, Any] = {}
        selection_resolver = getattr(child_agent, "_get_llm_selection", None)
        if callable(selection_resolver):
            try:
                selection = selection_resolver()
                if isinstance(selection, dict):
                    child_selection = {
                        str(key): value for key, value in selection.items()
                    }
            except Exception:
                child_selection = {}
        if not child_selection:
            child_selection = {
                "tier": str(getattr(child_agent, "tier", "normal") or "normal").strip().lower()
            }
        effective_agent_config = getattr(child_agent, "agent_config", child_agent_config)
        child_allowlist = self._derive_sub_agent_allowlist(
            child_registry=child_registry,
            child_agent=child_agent,
            params=normalized_params,
            child_sandbox=child_sandbox,
        )

        child_harness = AgentHarness(
            effective_agent_config,
            child_registry,
            self.llm_registry,
            child_sandbox,
            agent_name=normalized_name,
            tool_allowlist=child_allowlist,
            lsm_config=self.lsm_config,
            llm_service=child_selection.get("service"),
            llm_tier=child_selection.get("tier"),
            llm_provider=child_selection.get("provider"),
            llm_model=child_selection.get("model"),
            llm_temperature=child_selection.get("temperature"),
            llm_max_tokens=child_selection.get("max_tokens"),
            vectordb_config=self.vectordb_config,
            memory_store=self.memory_store,
            memory_context_builder=self.memory_context_builder,
            interaction_channel=self.interaction_channel,
        )
        topic = str(
            normalized_params.get("topic", normalized_params.get("goal", ""))
        ).strip() or f"Sub-agent run: {normalized_name}"
        context = AgentContext(
            messages=[{"role": "user", "content": topic}],
            budget_tracking={
                "tokens_used": 0,
                "max_tokens_budget": int(effective_agent_config.max_tokens_budget),
                "iterations": 0,
                "started_at": datetime.utcnow().isoformat(),
                "spawned_by": self.agent_name,
                "spawn_parent_state": str(self._state_path) if self._state_path else None,
            },
        )

        thread = threading.Thread(
            target=self._run_sub_agent_thread,
            args=(agent_id, child_harness, context),
            daemon=True,
            name=f"SubAgent-{agent_id}",
        )
        run_record = _SubAgentRun(
            agent_id=agent_id,
            agent_name=normalized_name,
            params=normalized_params,
            harness=child_harness,
            thread=thread,
            created_at=datetime.utcnow(),
        )
        with self._lock:
            self._sub_agent_runs[agent_id] = run_record

        thread.start()
        self.state.add_artifact(str(sub_root))
        return {
            "agent_id": agent_id,
            "agent_name": normalized_name,
            "status": "running",
            "sub_agent_root": str(sub_root),
            "topic": topic,
        }

    def await_sub_agent(
        self,
        agent_id: str,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Wait for a spawned sub-agent run to finish.

        Args:
            agent_id: Spawned run identifier.
            timeout_seconds: Optional wait timeout.

        Returns:
            Completion payload with status and artifact summary.
        """
        run = self._get_sub_agent_run(agent_id)
        timeout = None if timeout_seconds is None else max(0.001, float(timeout_seconds))
        run.thread.join(timeout=timeout)
        done = not run.thread.is_alive()

        state = run.harness.state
        status_value = (
            str(state.status.value) if hasattr(state.status, "value") else str(state.status)
        )
        if not done:
            status_value = "running"

        artifacts = list(state.artifacts) if done else []
        if done:
            for artifact in artifacts:
                self.state.add_artifact(str(artifact))

        return {
            "agent_id": run.agent_id,
            "agent_name": run.agent_name,
            "done": done,
            "status": status_value,
            "error": run.error,
            "created_at": run.created_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "artifacts": artifacts,
            "state_path": (
                str(run.harness.get_state_path())
                if run.harness.get_state_path() is not None
                else None
            ),
        }

    def collect_sub_agent_artifacts(self, agent_id: str, *, pattern: str = "*") -> list[str]:
        """
        Collect artifacts produced by a spawned sub-agent.

        Args:
            agent_id: Spawned run identifier.
            pattern: Optional glob for filtering artifact paths.

        Returns:
            Matching artifact paths.
        """
        run = self._get_sub_agent_run(agent_id)
        glob = str(pattern or "*").strip() or "*"
        artifacts = list(run.harness.state.artifacts)
        filtered: list[str] = []
        for artifact in artifacts:
            value = str(artifact).strip()
            if not value:
                continue
            path = Path(value)
            if fnmatch.fnmatch(value, glob) or fnmatch.fnmatch(path.name, glob):
                filtered.append(value)
                self.state.add_artifact(value)
        return filtered

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
                    "prompt": redact_secrets(entry.prompt) if entry.prompt else None,
                    "raw_response": redact_secrets(entry.raw_response) if entry.raw_response else None,
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
        if self._agent_workspace_root is None:
            self._agent_workspace_root = ensure_agent_workspace(
                self.agent_name,
                self.agent_config.agents_folder,
                sandbox=self.sandbox,
            )
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.agent_name}_{timestamp}"
        run_root_base = self._agent_workspace_root / "logs"
        self._run_root = run_root_base / run_name
        self._workspace_path = self._run_root / "workspace"
        self._workspace_path.mkdir(parents=True, exist_ok=True)
        filename = f"{run_name}_state.json"
        self._state_path = self._run_root / filename
        self.state.add_artifact(str(self._workspace_path))

    def _ensure_agent_workspace(self) -> Path:
        if self._agent_workspace_root is None:
            self._agent_workspace_root = ensure_agent_workspace(
                self.agent_name,
                self.agent_config.agents_folder,
                sandbox=self.sandbox,
            )
        return self._agent_workspace_root

    def _append_log(self, entry: AgentLogEntry) -> None:
        entry.content = redact_secrets(entry.content)
        if entry.prompt:
            entry.prompt = redact_secrets(entry.prompt)
        if entry.raw_response:
            entry.raw_response = redact_secrets(entry.raw_response)
        self.state.add_log(entry)
        if self.log_callback is None:
            return
        try:
            self.log_callback(entry)
        except Exception:
            logger.exception(
                "AgentHarness log callback failed for agent '%s'",
                self.agent_name,
            )

    def _parse_tool_response(self, raw_response: str) -> ToolResponse:
        text = str(raw_response or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ToolResponse(response=text, action=None, action_arguments={})

        if not isinstance(parsed, dict):
            return ToolResponse(response=text, action=None, action_arguments={})

        tool_calls = parsed.get("tool_calls") or parsed.get("tool_call")
        if tool_calls:
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if isinstance(tool_calls, list) and tool_calls:
                first_call = tool_calls[0]
                action = (
                    str(first_call.get("name")).strip()
                    if isinstance(first_call, dict) and first_call.get("name") is not None
                    else None
                )
                arguments = {}
                if isinstance(first_call, dict):
                    arguments = first_call.get("arguments") or first_call.get("args") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                return ToolResponse(
                    response=str(parsed.get("response", "")),
                    action=action,
                    action_arguments=arguments,
                )

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
        include_response_instruction: bool = True,
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
        if include_response_instruction:
            lines.append(
                "Respond with strict JSON: "
                '{"response":"...","action":"TOOL_OR_DONE","action_arguments":{}}'
            )
        return "\n".join(lines)


    @staticmethod
    def _format_tool_definitions_for_prompt(
        tool_definitions: list[Dict[str, Any]],
    ) -> str:
        formatted: list[Dict[str, Any]] = []
        for definition in tool_definitions:
            formatted.append(
                {
                    "name": definition.get("name"),
                    "description": definition.get("description"),
                    "parameters": definition.get("input_schema", {}),
                }
            )
        return json.dumps(formatted, indent=2)

    def _build_system_prompt(
        self,
        tool_definitions: list[Dict[str, Any]],
        *,
        use_function_calling: bool,
    ) -> str:
        lines = [
            "You are an agent that can call tools to complete tasks.",
            "If no tool is required, respond directly.",
        ]
        if self.system_prompt:
            lines.append(self.system_prompt)
        if use_function_calling:
            lines.append("Use the provided tool schema when calling tools.")
            return "\n".join(lines)

        if tool_definitions:
            lines.append("Available tools (function calling schema):")
            lines.append(self._format_tool_definitions_for_prompt(tool_definitions))
        return "\n".join(lines)

    def _call_llm(
        self,
        llm_provider: Any,
        *,
        messages_for_llm: list[Dict[str, Any]],
        standing_context_block: str,
        tool_definitions: Optional[list[Dict[str, Any]]] = None,
    ) -> tuple[str, str, str]:
        if tool_definitions is None:
            tool_definitions = (
                self.context.tool_definitions
                if self.context is not None
                else self._list_tool_definitions()
            )
        supports_function_calling = bool(
            getattr(llm_provider, "supports_function_calling", False)
        )
        system_prompt = self._build_system_prompt(
            tool_definitions,
            use_function_calling=supports_function_calling,
        )
        user_prompt = self._messages_to_prompt(
            messages_for_llm,
            standing_context_block=standing_context_block,
            include_response_instruction=not supports_function_calling,
        )
        temperature = None
        max_tokens = None
        temperature_resolver = getattr(llm_provider, "_default_temperature", None)
        max_tokens_resolver = getattr(llm_provider, "_default_max_tokens", None)
        if callable(temperature_resolver):
            temperature = temperature_resolver()
        if callable(max_tokens_resolver):
            max_tokens = max_tokens_resolver()
        if max_tokens is None:
            max_tokens = int(self.agent_config.max_tokens_budget)

        max_tokens_value = int(self.agent_config.max_tokens_budget)
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            max_tokens_value = max(1, max_tokens)
        elif isinstance(max_tokens, str):
            stripped = max_tokens.strip()
            if stripped.isdigit():
                max_tokens_value = max(1, int(stripped))

        llm_kwargs: dict[str, Any] = {}
        if supports_function_calling and tool_definitions:
            llm_kwargs["tools"] = tool_definitions
            llm_kwargs["tool_choice"] = "auto"

        raw_response = llm_provider.send_message(
            input=user_prompt,
            instruction=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens_value,
            **llm_kwargs,
        )
        return str(raw_response), system_prompt, user_prompt

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

    def _resolve_llm_config(self) -> Any:
        if self.llm_provider and self.llm_model:
            return self.llm_registry.resolve_direct(
                self.llm_provider,
                self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        if self.llm_service:
            return self.llm_registry.resolve_service(self.llm_service)
        if self.llm_tier:
            try:
                return self.llm_registry.resolve_tier(self.llm_tier)
            except Exception:
                warnings.warn(
                    f"LLM tier '{self.llm_tier}' is not configured; "
                    "falling back to 'default' service.",
                    stacklevel=2,
                )
                return self.llm_registry.resolve_service("default")
        return self.llm_registry.resolve_service("default")

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

    def _write_agent_log(self) -> Optional[Path]:
        if not self.state.log_entries:
            return None
        verbosity = "normal"
        if isinstance(self.agent_config.agent_configs, dict):
            overrides = self.agent_config.agent_configs.get(self.agent_name, {})
            if isinstance(overrides, dict):
                verbosity = overrides.get("log_verbosity", verbosity)
        try:
            path = save_agent_log(
                self.state.log_entries,
                agent_name=self.agent_name,
                agents_folder=self.agent_config.agents_folder,
                verbosity=verbosity,
            )
            self.state.add_artifact(str(path))
            return path
        except Exception:
            logger.exception(
                "Failed to persist agent log for '%s'",
                self.agent_name,
            )
            return None

    def _bind_runtime_tools(self) -> None:
        for tool in self.tool_registry.list_tools():
            binder = getattr(tool, "bind_harness", None)
            if callable(binder):
                binder(self)

    def _run_sub_agent_thread(
        self,
        agent_id: str,
        sub_harness: AgentHarness,
        context: AgentContext,
    ) -> None:
        error: Optional[str] = None
        try:
            sub_harness.run(context)
        except Exception as exc:
            error = str(exc)
            logger.exception("Spawned sub-agent run '%s' failed", agent_id)
        finally:
            with self._lock:
                run = self._sub_agent_runs.get(agent_id)
                if run is not None:
                    run.completed_at = datetime.utcnow()
                    run.error = error

    def _get_sub_agent_run(self, agent_id: str) -> _SubAgentRun:
        normalized = str(agent_id or "").strip()
        if not normalized:
            raise ValueError("agent_id is required")
        with self._lock:
            run = self._sub_agent_runs.get(normalized)
        if run is None:
            raise KeyError(f"Unknown sub-agent run id: {normalized}")
        return run

    def _build_sub_agent_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in self.tool_registry.list_tools():
            if str(tool.name).strip() in self._meta_tool_names:
                continue
            registry.register(tool)
        return registry

    def _derive_sub_agent_allowlist(
        self,
        *,
        child_registry: ToolRegistry,
        child_agent: Any,
        params: Dict[str, Any],
        child_sandbox: ToolSandbox,
    ) -> set[str]:
        allow_network = bool(params.get("allow_network", False))
        allow_exec = bool(params.get("allow_exec", False))
        allowed_risks = {"read_only", "writes_workspace"}
        if allow_network and child_sandbox.config.allow_url_access:
            allowed_risks.add("network")
        if allow_exec:
            allowed_risks.add("exec")

        allowlist = {
            str(definition.get("name", "")).strip()
            for definition in child_registry.list_definitions()
            if str(definition.get("name", "")).strip()
            and str(definition.get("risk_level", "read_only")).strip() in allowed_risks
        }

        agent_allowlist = getattr(child_agent, "tool_allowlist", None)
        if agent_allowlist:
            normalized = {
                str(name).strip() for name in agent_allowlist if str(name).strip()
            }
            allowlist &= normalized

        remote_sources = getattr(child_agent, "remote_source_allowlist", None)
        if remote_sources is not None:
            allowed_remote = {f"query_{src}" for src in remote_sources}
            remote_tools = {
                name
                for name in allowlist
                if name.startswith("query_") and name not in self._BUILTIN_QUERY_TOOL_NAMES
            }
            allowlist -= (remote_tools - allowed_remote)

        explicit_allowlist = params.get("tool_allowlist")
        if isinstance(explicit_allowlist, list):
            requested = {
                str(name).strip() for name in explicit_allowlist if str(name).strip()
            }
            allowlist &= requested

        if any(str(tool.name).strip() == "ask_user" for tool in child_registry.list_tools()):
            allowlist.add("ask_user")

        return allowlist

    def _build_sub_agent_sandbox(
        self,
        *,
        sub_root: Path,
        shared_workspace: Path,
        params: Dict[str, Any],
    ) -> ToolSandbox:
        parent = self.sandbox.config
        sub_root = resolve_path(sub_root, strict=False)
        shared_workspace = resolve_path(shared_workspace, strict=False)

        if not self._path_allowed_by_parent(sub_root, action="write"):
            raise PermissionError(
                f"Sub-agent workspace '{sub_root}' is outside parent writable sandbox paths"
            )
        if not self._path_allowed_by_parent(sub_root, action="read"):
            raise PermissionError(
                f"Sub-agent workspace '{sub_root}' is outside parent readable sandbox paths"
            )

        read_paths: list[Path] = []
        for candidate in self.sandbox.effective_read_paths():
            if self._path_allowed_by_parent(candidate, action="read"):
                read_paths.append(candidate)
        if self._path_allowed_by_parent(shared_workspace, action="read"):
            read_paths.append(shared_workspace)
        if self._path_allowed_by_parent(sub_root, action="read"):
            read_paths.append(sub_root)

        # Preserve declaration order while deduplicating.
        deduped_read_paths: list[Path] = []
        seen_read: set[str] = set()
        for path in read_paths:
            marker = str(path)
            if marker in seen_read:
                continue
            seen_read.add(marker)
            deduped_read_paths.append(path)
        if not deduped_read_paths:
            raise PermissionError("Parent sandbox provides no readable paths for sub-agent run")

        allow_network = bool(params.get("allow_network", False)) or bool(
            params.get("allow_url_access", False)
        )
        force_docker = parent.force_docker or bool(params.get("force_docker", False))
        execution_mode = "local_only"
        if parent.execution_mode == "prefer_docker" or force_docker or allow_network:
            execution_mode = "prefer_docker"

        child_config = SandboxConfig(
            allowed_read_paths=deduped_read_paths,
            allowed_write_paths=[sub_root],
            allow_url_access=bool(parent.allow_url_access and allow_network),
            require_user_permission=dict(parent.require_user_permission),
            require_permission_by_risk=dict(parent.require_permission_by_risk),
            execution_mode=execution_mode,
            force_docker=force_docker,
            limits=dict(parent.limits),
            docker=dict(parent.docker),
            tool_llm_assignments=dict(parent.tool_llm_assignments),
        )
        return ToolSandbox(
            child_config,
            global_sandbox=parent,
            local_runner=self.sandbox.local_runner,
            docker_runner=self.sandbox.docker_runner,
            interaction_channel=self.interaction_channel,
        )

    def _path_allowed_by_parent(self, path: Path, *, action: str) -> bool:
        try:
            if action == "read":
                self.sandbox.check_read_path(path)
            else:
                self.sandbox.check_write_path(path)
            return True
        except (PermissionError, ValueError):
            return False

    @staticmethod
    def _normalize_allowlist(tool_allowlist: Optional[Set[str]]) -> Optional[Set[str]]:
        if tool_allowlist is None:
            return None
        normalized = {str(name).strip() for name in tool_allowlist if str(name).strip()}
        normalized.add("ask_user")
        return normalized

    @staticmethod
    def _normalize_remote_sources(
        remote_source_allowlist: Optional[Set[str]],
    ) -> Optional[Set[str]]:
        if remote_source_allowlist is None:
            return None
        return {
            str(name).strip()
            for name in remote_source_allowlist
            if str(name).strip()
        }

    def _resolve_allowed_tool_names(
        self,
        *,
        tool_names: Optional[set[str]] = None,
    ) -> set[str]:
        registered = {
            str(tool.name).strip()
            for tool in self.tool_registry.list_tools()
            if str(tool.name).strip()
        }
        if self.tool_allowlist is None:
            allowed = set(registered)
        else:
            allowed = {name for name in (self.tool_allowlist or set()) if name}
            allowed |= set(self._ALWAYS_AVAILABLE_TOOLS)
            allowed &= registered
        if tool_names is not None:
            allowed &= {str(name).strip() for name in tool_names if str(name).strip()}

        if self.remote_source_allowlist is not None:
            allowed_remote = {f"query_{src}" for src in self.remote_source_allowlist}
            remote_tools = {
                name for name in allowed
                if name.startswith("query_") and name not in self._BUILTIN_QUERY_TOOL_NAMES
            }
            allowed -= (remote_tools - allowed_remote)

        sandbox_config = getattr(self.sandbox, "config", None)
        if sandbox_config is None:
            sandbox_config = getattr(self.agent_config, "sandbox", None)
        if sandbox_config is not None:
            allow_url_access = bool(getattr(sandbox_config, "allow_url_access", False))
            if not allow_url_access:
                network_tools = set(ToolSandbox._NETWORK_TOOL_NAMES)
                network_tools |= {
                    name
                    for name in allowed
                    if name.startswith("query_")
                    and name not in self._BUILTIN_QUERY_TOOL_NAMES
                }
                allowed -= network_tools
            if not list(getattr(sandbox_config, "allowed_read_paths", []) or []):
                allowed -= set(ToolSandbox._READ_TOOL_NAMES)
            if not list(getattr(sandbox_config, "allowed_write_paths", []) or []):
                allowed -= set(ToolSandbox._WRITE_TOOL_NAMES)

        return {name for name in allowed if name}

    def _list_tool_definitions(
        self,
        *,
        tool_names: Optional[set[str]] = None,
    ) -> list[Dict[str, Any]]:
        definitions = self.tool_registry.list_definitions()
        allowed = self._resolve_allowed_tool_names(tool_names=tool_names)
        if not allowed:
            return []
        return [
            definition
            for definition in definitions
            if str(definition.get("name", "")).strip() in allowed
        ]

    def _is_tool_allowed(self, tool_name: str) -> bool:
        normalized = str(tool_name).strip()
        if not normalized:
            return False
        return normalized in self._resolve_allowed_tool_names()

    def request_interaction(self, request: InteractionRequest) -> InteractionResponse:
        """
        Post an interaction request and block for a user response.
        """
        if self.interaction_channel is None:
            raise RuntimeError("Interaction channel is not configured for this harness")
        self._set_waiting_for_user(True)
        try:
            return self.interaction_channel.post_request(request)
        finally:
            self._set_waiting_for_user(False)

    def _set_waiting_for_user(self, waiting: bool) -> None:
        if waiting:
            if self.state.status == AgentStatus.RUNNING:
                self.state.set_status(AgentStatus.WAITING_USER)
                self.save_state()
            return
        if self.state.status == AgentStatus.WAITING_USER:
            if self._stop_event.is_set():
                self.state.set_status(AgentStatus.COMPLETED)
            else:
                self.state.set_status(AgentStatus.RUNNING)
            self.save_state()

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
