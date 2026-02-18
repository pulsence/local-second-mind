"""
Shell-level agent command helpers.
"""

from __future__ import annotations

from collections import deque
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Literal
from uuid import uuid4

from lsm.agents import (
    AgentHarness,
    AgentScheduler,
    InteractionChannel,
    InteractionRequest,
    InteractionResponse,
    create_agent,
)
from lsm.agents.log_formatter import format_agent_log
from lsm.agents.memory import Memory, MemoryCandidate, create_memory_store
from lsm.agents.memory.models import now_utc
from lsm.agents.models import AgentContext, AgentLogEntry
from lsm.agents.tools import ToolSandbox, create_default_tool_registry
from lsm.config.loader import save_config_to_file
from lsm.config.models import ScheduleConfig
from lsm.logging import get_logger
from lsm.ui.helpers.commands.common import (
    CommandParseError,
    format_command_error,
    tokenize_command,
)

logger = get_logger(__name__)

AgentRuntimeUIEventType = Literal[
    "run_started",
    "run_completed",
    "interaction_response",
]


@dataclass(frozen=True)
class AgentRuntimeUIEvent:
    """Typed runtime event emitted by manager threads for UI consumers."""

    event_type: AgentRuntimeUIEventType
    agent_id: str
    agent_name: str
    topic: str = ""
    status: str = ""
    request_id: str = ""
    request_type: str = ""
    decision: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentRunEntry:
    """
    Runtime record for a single agent run.
    """

    agent_id: str
    agent_name: str
    agent: Any
    thread: threading.Thread
    harness: Any
    channel: InteractionChannel
    started_at: datetime
    topic: str
    context: Optional[AgentContext] = None
    completed_at: Optional[datetime] = None


@dataclass
class _AgentLogStream:
    """
    Thread-safe buffered stream state for a single agent's live logs.
    """

    max_entries: int
    entries: deque[dict[str, Any]] = field(default_factory=deque)
    dropped_count: int = 0


class _NoopHarness:
    """
    Fallback harness used when the full harness cannot be initialized.
    """

    def stop(self) -> None:
        return

    def pause(self) -> None:
        return

    def resume(self) -> None:
        return


class AgentRuntimeManager:
    """
    Manage active agent runtime instances for UI command surfaces.
    """

    _DEFAULT_LOG_STREAM_QUEUE_LIMIT = 500

    def __init__(self, *, completed_retention: int = 10, join_timeout_s: float = 5.0) -> None:
        self._agents: dict[str, AgentRunEntry] = {}
        self._completed_runs: dict[str, AgentRunEntry] = {}
        self._completed_order: list[str] = []
        self._log_streams: dict[str, _AgentLogStream] = {}
        self._selected_agent_id: Optional[str] = None
        self._session_tool_approvals: set[str] = set()
        self._completed_retention = max(1, int(completed_retention))
        self._join_timeout_s = max(0.1, float(join_timeout_s))
        self._scheduler: Optional[AgentScheduler] = None
        self._ui_event_sink: Optional[Callable[[AgentRuntimeUIEvent], None]] = None
        self._lock = threading.RLock()

    def set_ui_event_sink(
        self,
        sink: Optional[Callable[[AgentRuntimeUIEvent], None]],
    ) -> None:
        """Set callback for typed runtime UI events."""
        with self._lock:
            self._ui_event_sink = sink

    def start(self, app: Any, agent_name: str, topic: str) -> str:
        with self._lock:
            self._cleanup_finished_locked()

            agent_cfg = getattr(app.config, "agents", None)
            if agent_cfg is None or not getattr(agent_cfg, "enabled", False):
                return "Agents are disabled. Enable `agents.enabled` in config.\n"

            max_concurrent = self._resolve_max_concurrent(app)
            if len(self._agents) >= max_concurrent:
                return (
                    f"Cannot start agent '{agent_name}': max_concurrent limit "
                    f"({max_concurrent}) reached.\n"
                )

            normalized_agent_name = str(agent_name or "").strip().lower()
            if not normalized_agent_name:
                return "Usage: /agent start <name> <topic>\n"
            normalized_topic = str(topic or "").strip()
            if not normalized_topic:
                return "Usage: /agent start <name> <topic>\n"

            interaction_cfg = getattr(agent_cfg, "interaction", None)
            timeout_seconds = int(getattr(interaction_cfg, "timeout_seconds", 300))
            timeout_action = str(getattr(interaction_cfg, "timeout_action", "deny"))
            channel = InteractionChannel(
                timeout_seconds=timeout_seconds,
                timeout_action=timeout_action,
            )
            for approved_tool in sorted(self._session_tool_approvals):
                try:
                    channel.approve_for_session(approved_tool)
                except Exception:
                    logger.exception(
                        "Failed to seed session approval '%s' into interaction channel",
                        approved_tool,
                    )

            memory_store = None
            if getattr(agent_cfg, "memory", None) is not None and agent_cfg.memory.enabled:
                try:
                    memory_store = create_memory_store(agent_cfg, app.config.vectordb)
                except Exception as exc:
                    return f"Failed to initialize agent memory store: {exc}\n"

            tool_registry = create_default_tool_registry(
                app.config,
                collection=getattr(app, "query_provider", None),
                embedder=getattr(app, "query_embedder", None),
                batch_size=getattr(app.config, "batch_size", 32),
                memory_store=memory_store,
            )
            sandbox = ToolSandbox(agent_cfg.sandbox)
            self._attach_interaction_channel_to_sandbox(
                sandbox=sandbox,
                channel=channel,
                topic=normalized_topic,
            )
            agent = create_agent(
                name=normalized_agent_name,
                llm_registry=app.config.llm,
                tool_registry=tool_registry,
                sandbox=sandbox,
                agent_config=agent_cfg,
            )
            agent_id = uuid4().hex
            log_queue_limit = self._resolve_log_stream_queue_limit(app)
            self._log_streams[agent_id] = _AgentLogStream(
                max_entries=log_queue_limit,
            )

            def _log_callback(entry: AgentLogEntry) -> None:
                self._enqueue_log_stream_entry(agent_id, entry)

            harness = self._build_harness(
                app=app,
                sandbox=sandbox,
                agent_cfg=agent_cfg,
                agent_name=normalized_agent_name,
                tool_registry=tool_registry,
                agent=agent,
                memory_store=memory_store,
                interaction_channel=channel,
                log_callback=_log_callback,
            )

            context = AgentContext(
                messages=[{"role": "user", "content": normalized_topic}],
                tool_definitions=tool_registry.list_definitions(),
                budget_tracking={
                    "tokens_used": 0,
                    "max_tokens_budget": agent_cfg.max_tokens_budget,
                    "started_at": datetime.utcnow().isoformat(),
                },
            )

            def _run() -> None:
                try:
                    agent.run(context)
                except Exception:
                    logger.exception(
                        "Agent run '%s' (%s) failed",
                        normalized_agent_name,
                        agent_id,
                    )
                finally:
                    if memory_store is not None:
                        try:
                            memory_store.close()
                        except Exception:
                            logger.exception(
                                "Failed to close memory store for run '%s' (%s)",
                                normalized_agent_name,
                                agent_id,
                            )
                    self._mark_run_completed(agent_id)

            thread = threading.Thread(
                target=_run,
                daemon=True,
                name=f"Agent-{normalized_agent_name}-{agent_id[:8]}",
            )
            self._agents[agent_id] = AgentRunEntry(
                agent_id=agent_id,
                agent_name=normalized_agent_name,
                agent=agent,
                thread=thread,
                harness=harness,
                channel=channel,
                started_at=datetime.utcnow(),
                topic=normalized_topic,
                context=context,
            )
            self._attach_log_stream_to_agent(
                agent_id=agent_id,
                agent=agent,
            )
            self._selected_agent_id = agent_id
            thread.start()
            self._emit_ui_event(
                AgentRuntimeUIEvent(
                    event_type="run_started",
                    agent_id=agent_id,
                    agent_name=normalized_agent_name,
                    topic=normalized_topic,
                    status="running",
                )
            )
            return (
                f"Started agent '{normalized_agent_name}' "
                f"(id={agent_id}) with topic: {normalized_topic}\n"
            )

    def status(self, agent_id: Optional[str] = None) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            if agent_id is not None:
                try:
                    entry = self._lookup_entry_locked(agent_id, include_completed=True)
                except ValueError as exc:
                    return f"{exc}\n"
                if entry is None:
                    return f"Unknown agent id: {agent_id}\n"
                return self._format_entry_status(entry)

            active_entries = sorted(self._agents.values(), key=lambda item: item.started_at)
            completed_entries = [
                self._completed_runs[item_id]
                for item_id in self._completed_order
                if item_id in self._completed_runs
            ]

        if not active_entries and not completed_entries:
            return "No active agent.\n"
        if len(active_entries) == 1 and not completed_entries:
            return self._format_entry_status(active_entries[0])
        if not active_entries and len(completed_entries) == 1:
            return self._format_entry_status(completed_entries[0])

        lines = [
            f"Agents: {len(active_entries)} active, {len(completed_entries)} recent completed",
        ]
        if active_entries:
            lines.append("Active:")
            for entry in active_entries:
                lines.append(
                    self._format_compact_entry_status(entry, include_topic=True)
                )
        if completed_entries:
            lines.append("Recent Completed:")
            for entry in reversed(completed_entries):
                lines.append(
                    self._format_compact_entry_status(entry, include_topic=True)
                )
        lines.append("")
        return "\n".join(lines)

    def select(self, agent_id: str) -> str:
        normalized = str(agent_id or "").strip()
        if not normalized:
            return "Usage: /agent select <agent_id>\n"
        with self._lock:
            self._cleanup_finished_locked()
            try:
                entry = self._lookup_entry_locked(normalized, include_completed=True)
            except ValueError as exc:
                return f"{exc}\n"
            if entry is None:
                return f"Unknown agent id: {normalized}\n"
            self._selected_agent_id = entry.agent_id
            is_running = entry.agent_id in self._agents and entry.thread.is_alive()
        state = "running" if is_running else "completed"
        return f"Selected agent '{entry.agent_name}' ({entry.agent_id}) [{state}].\n"

    def format_running_agents(self) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            entries = sorted(self._agents.values(), key=lambda item: item.started_at)
            selected_id = self._selected_agent_id
        if not entries:
            return "No running agents.\n"
        lines = [f"Running agents ({len(entries)}):"]
        now = datetime.utcnow()
        for entry in entries:
            marker = "*" if entry.agent_id == selected_id else "-"
            status = self._entry_status_value(entry)
            age_seconds = max(0.0, (now - entry.started_at).total_seconds())
            lines.append(
                f"{marker} {entry.agent_id[:8]} | {entry.agent_name} "
                f"| status={status} | age={round(age_seconds, 1)}s | topic={entry.topic}"
            )
        lines.append("")
        return "\n".join(lines)

    def format_pending_interactions(self, agent_id: Optional[str] = None) -> str:
        pending = self.get_pending_interactions(agent_id=agent_id)
        if not pending:
            return "No pending interaction requests.\n"
        lines = [f"Pending interactions ({len(pending)}):"]
        for row in pending:
            agent_id_value = str(row.get("agent_id", "")).strip()
            agent_name = str(row.get("agent_name", "")).strip() or "-"
            request_type = str(row.get("request_type", "")).strip() or "-"
            request_id = str(row.get("request_id", "")).strip() or "-"
            tool_name = str(row.get("tool_name", "")).strip() or "-"
            risk_level = str(row.get("risk_level", "")).strip() or "-"
            lines.append(
                f"- {agent_id_value[:8]} | {agent_name} | type={request_type} "
                f"| request_id={request_id} | tool={tool_name} | risk={risk_level}"
            )
            reason = str(row.get("reason", "")).strip()
            if reason:
                lines.append(f"  reason: {reason}")
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                lines.append(f"  prompt: {prompt}")
        lines.append("")
        return "\n".join(lines)

    def start_meta(self, app: Any, goal: str) -> str:
        normalized_goal = str(goal or "").strip()
        if not normalized_goal:
            return "Usage: /agent meta start <goal>\n"
        return self.start(app, "meta", normalized_goal)

    def get_meta_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_finished_locked()
            entry = self._latest_entry_for_agent_name_locked("meta")

        if entry is None:
            return self._empty_meta_snapshot()

        agent = entry.agent
        alive = bool(entry.thread.is_alive())
        status = str(getattr(agent.state.status, "value", agent.state.status))
        result = getattr(agent, "last_result", None)
        graph = getattr(result, "task_graph", None)

        tasks: list[dict[str, Any]] = []
        if graph is not None:
            for task in getattr(graph, "tasks", []):
                tasks.append(
                    {
                        "id": str(getattr(task, "id", "")),
                        "agent_name": str(getattr(task, "agent_name", "")),
                        "status": str(getattr(task, "status", "")),
                        "depends_on": list(getattr(task, "depends_on", [])),
                    }
                )

        task_runs: list[dict[str, Any]] = []
        for run in list(getattr(result, "task_runs", []) if result is not None else []):
            task_runs.append(
                {
                    "task_id": str(getattr(run, "task_id", "")),
                    "agent_name": str(getattr(run, "agent_name", "")),
                    "status": str(getattr(run, "status", "")),
                    "sub_agent_dir": str(getattr(run, "sub_agent_dir", "")),
                    "artifacts": list(getattr(run, "artifacts", [])),
                    "error": getattr(run, "error", None),
                }
            )

        return {
            "available": True,
            "status": status,
            "thread_alive": alive,
            "goal": str(getattr(result, "goal", "")),
            "execution_order": list(
                getattr(result, "execution_order", getattr(agent, "last_execution_order", []))
            ),
            "tasks": tasks,
            "task_runs": task_runs,
            "artifacts": list(getattr(agent.state, "artifacts", [])),
            "final_result_path": str(getattr(result, "final_result_path", "") or ""),
            "meta_log_path": str(getattr(result, "meta_log_path", "") or ""),
        }

    def format_meta_status(self) -> str:
        snapshot = self.get_meta_snapshot()
        if not snapshot.get("available"):
            return "No active meta-agent.\n"

        tasks = list(snapshot.get("tasks", []))
        completed = sum(1 for task in tasks if str(task.get("status")) == "completed")
        failed = sum(1 for task in tasks if str(task.get("status")) == "failed")
        lines = [
            "Meta Agent Status:",
            f"- Status: {snapshot.get('status')}",
            f"- Thread alive: {snapshot.get('thread_alive')}",
        ]
        goal = str(snapshot.get("goal") or "").strip()
        if goal:
            lines.append(f"- Goal: {goal}")
        lines.append(
            f"- Tasks: {len(tasks)} total, {completed} completed, {failed} failed"
        )
        execution_order = list(snapshot.get("execution_order", []))
        if execution_order:
            lines.append("- Execution order: " + ", ".join(str(item) for item in execution_order))
        task_runs = list(snapshot.get("task_runs", []))
        if task_runs:
            lines.append("- Sub-agent runs:")
            for run in task_runs:
                lines.append(
                    f"  {run.get('task_id')} ({run.get('agent_name')}): "
                    f"{run.get('status')} artifacts={len(run.get('artifacts', []))}"
                )
        lines.append("")
        return "\n".join(lines)

    def meta_log(self) -> str:
        snapshot = self.get_meta_snapshot()
        if not snapshot.get("available"):
            return "No active meta-agent.\n"

        meta_log_path = str(snapshot.get("meta_log_path") or "").strip()
        if meta_log_path:
            path = Path(meta_log_path)
            if path.exists() and path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    return f"Failed to read meta log '{path}': {exc}\n"
                if text.strip():
                    return text if text.endswith("\n") else text + "\n"
        return self.log(agent_id=None, prefer_meta=True)

    def stop(self, agent_id: Optional[str] = None) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            entry, error = self._resolve_target_entry_locked(
                agent_id=agent_id,
                action="stop",
                include_completed=True,
            )
            if error:
                return error
            assert entry is not None
            thread = entry.thread
            target_id = entry.agent_id

        self._cancel_entry_pending_interactions(
            entry,
            reason="Agent stopped",
            close_channel=True,
        )
        self._safe_call(entry.harness, "stop")
        self._safe_call(entry.agent, "stop")

        if thread.is_alive():
            stop_wait_timeout = max(self._join_timeout_s, 30.0)
            self._wait_for_thread_completion(
                thread,
                timeout_s=stop_wait_timeout,
            )

        with self._lock:
            self._cleanup_finished_locked()
            still_running = (
                target_id in self._agents
                and self._agents[target_id].thread.is_alive()
            )
        if still_running:
            return (
                f"Stop requested for agent '{entry.agent_name}' "
                f"({target_id}), but thread is still running.\n"
            )
        return f"Stop requested for agent '{entry.agent_name}' ({target_id}).\n"

    def pause(self, agent_id: Optional[str] = None) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            entry, error = self._resolve_target_entry_locked(
                agent_id=agent_id,
                action="pause",
                include_completed=True,
            )
            if error:
                return error
            assert entry is not None

        self._safe_call(entry.harness, "pause")
        self._safe_call(entry.agent, "pause")
        return f"Paused agent '{entry.agent_name}' ({entry.agent_id}).\n"

    def resume(self, agent_id: Optional[str] = None) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            entry, error = self._resolve_target_entry_locked(
                agent_id=agent_id,
                action="resume",
                include_completed=True,
            )
            if error:
                return error
            assert entry is not None

        self._safe_call(entry.harness, "resume")
        self._safe_call(entry.agent, "resume")
        return f"Resumed agent '{entry.agent_name}' ({entry.agent_id}).\n"

    def queue_user_command(
        self,
        message: str,
        *,
        agent_id: Optional[str] = None,
        resume_after_queue: bool = False,
    ) -> str:
        normalized_message = str(message or "").strip()
        if not normalized_message:
            return "Usage: /agent queue [agent_id] <message>\n"
        with self._lock:
            self._cleanup_finished_locked()
            entry, error = self._resolve_target_entry_locked(
                agent_id=agent_id,
                action="queue",
                include_completed=False,
            )
            if error:
                return error
            assert entry is not None
            if entry.context is None:
                return (
                    f"Cannot queue message for agent '{entry.agent_name}' "
                    f"({entry.agent_id}): run context unavailable.\n"
                )
            entry.context.messages.append(
                {"role": "user", "content": normalized_message}
            )
            state = getattr(entry.agent, "state", None)
            add_log = getattr(state, "add_log", None)
            if callable(add_log):
                try:
                    add_log(
                        AgentLogEntry(
                            timestamp=datetime.utcnow(),
                            actor="user",
                            content=normalized_message,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to append queued user message log for agent '%s' (%s)",
                        entry.agent_name,
                        entry.agent_id,
                    )

        if resume_after_queue:
            self._safe_call(entry.harness, "resume")
            self._safe_call(entry.agent, "resume")
            return (
                f"Queued user message for agent '{entry.agent_name}' "
                f"({entry.agent_id}).\n"
                f"Resumed agent '{entry.agent_name}' ({entry.agent_id}).\n"
            )
        return (
            f"Queued user message for agent '{entry.agent_name}' "
            f"({entry.agent_id}).\n"
        )

    def log(self, agent_id: Optional[str] = None, *, prefer_meta: bool = False) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            if prefer_meta:
                entry = self._latest_entry_for_agent_name_locked("meta")
                if entry is None:
                    return "No active meta-agent.\n"
            else:
                entry, error = self._resolve_target_entry_locked(
                    agent_id=agent_id,
                    action="log",
                    include_completed=True,
                )
                if error:
                    return error
                assert entry is not None

        entries = list(getattr(entry.agent.state, "log_entries", []))
        if not entries:
            return "No log entries yet.\n"
        return format_agent_log(entries)

    def get_log_entries(self, agent_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Return raw log entries for an agent as dicts for UI formatting.

        Args:
            agent_id: Optional agent ID to get entries for. If None, resolves
                to the selected or only running agent.

        Returns:
            List of dict entries with keys: actor, content, action, action_arguments.
        """
        with self._lock:
            self._cleanup_finished_locked()
            entry, error = self._resolve_target_entry_locked(
                agent_id=agent_id,
                action="log",
                include_completed=True,
            )
            if error or entry is None:
                return []

        raw_entries = list(getattr(entry.agent.state, "log_entries", []))
        result: list[dict[str, Any]] = []
        for e in raw_entries:
            result.append({
                "actor": getattr(e, "actor", ""),
                "content": getattr(e, "content", ""),
                "action": getattr(e, "action", None),
                "action_arguments": getattr(e, "action_arguments", None),
            })
        return result

    def get_active_agent(self):
        """Return the active agent instance for UI screens."""
        with self._lock:
            self._cleanup_finished_locked()
            if self._selected_agent_id and self._selected_agent_id in self._agents:
                return self._agents[self._selected_agent_id].agent
            if len(self._agents) == 1:
                return next(iter(self._agents.values())).agent
            return None

    def list_running(self) -> list[dict[str, Any]]:
        with self._lock:
            self._cleanup_finished_locked()
            entries = sorted(self._agents.values(), key=lambda item: item.started_at)
        now = datetime.utcnow()
        output: list[dict[str, Any]] = []
        for entry in entries:
            status = self._entry_status_value(entry)
            output.append(
                {
                    "agent_id": entry.agent_id,
                    "agent_name": entry.agent_name,
                    "topic": entry.topic,
                    "status": status,
                    "started_at": entry.started_at.isoformat(),
                    "duration_seconds": max(
                        0.0,
                        (now - entry.started_at).total_seconds(),
                    ),
                    "thread_alive": entry.thread.is_alive(),
                    "log_entries": len(getattr(entry.agent.state, "log_entries", [])),
                }
            )
        return output

    def list_completed(self) -> list[dict[str, Any]]:
        """Return list of completed agent runs from the current session.

        Returns:
            List of dict entries with keys: agent_id, agent_name, topic,
            status, started_at, completed_at, duration_seconds.
        """
        with self._lock:
            completed_entries = [self._completed_runs[k] for k in self._completed_order if k in self._completed_runs]
            entries = sorted(completed_entries, key=lambda item: item.started_at, reverse=True)
        output: list[dict[str, Any]] = []
        for entry in entries:
            status = self._entry_status_value(entry)
            completed_at = entry.completed_at
            started_at = entry.started_at
            duration_seconds = 0.0
            if completed_at and started_at:
                duration_seconds = (completed_at - started_at).total_seconds()
            output.append(
                {
                    "agent_id": entry.agent_id,
                    "agent_name": entry.agent_name,
                    "topic": entry.topic,
                    "status": status,
                    "started_at": started_at.isoformat() if started_at else "",
                    "completed_at": completed_at.isoformat() if completed_at else "",
                    "duration_seconds": duration_seconds,
                    "thread_alive": False,
                    "log_entries": len(getattr(entry.agent.state, "log_entries", [])),
                }
            )
        return output

    def drain_log_stream(
        self,
        agent_id: str,
        *,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        normalized = str(agent_id or "").strip()
        if not normalized:
            return {"agent_id": "", "entries": [], "dropped_count": 0, "has_more": False}
        limit = max(1, int(max_entries))
        with self._lock:
            self._cleanup_finished_locked()
            try:
                entry = self._lookup_entry_locked(normalized, include_completed=True)
            except ValueError as exc:
                return {
                    "agent_id": normalized,
                    "entries": [],
                    "dropped_count": 0,
                    "has_more": False,
                    "error": str(exc),
                }
            if entry is None:
                return {"agent_id": normalized, "entries": [], "dropped_count": 0, "has_more": False}

            stream = self._log_streams.get(entry.agent_id)
            if stream is None:
                return {"agent_id": entry.agent_id, "entries": [], "dropped_count": 0, "has_more": False}

            drained: list[dict[str, Any]] = []
            while stream.entries and len(drained) < limit:
                drained.append(stream.entries.popleft())
            dropped = int(stream.dropped_count)
            stream.dropped_count = 0
            return {
                "agent_id": entry.agent_id,
                "entries": drained,
                "dropped_count": dropped,
                "has_more": bool(stream.entries),
            }

    def clear_log_stream(self, agent_id: str) -> None:
        normalized = str(agent_id or "").strip()
        if not normalized:
            return
        with self._lock:
            try:
                entry = self._lookup_entry_locked(normalized, include_completed=True)
            except ValueError:
                return
            if entry is None:
                return
            stream = self._log_streams.get(entry.agent_id)
            if stream is None:
                return
            stream.entries.clear()
            stream.dropped_count = 0

    def get_pending_interactions(
        self,
        agent_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._cleanup_finished_locked()
            if agent_id is None:
                entries = list(self._agents.values())
            else:
                try:
                    entry = self._lookup_entry_locked(agent_id, include_completed=False)
                except ValueError:
                    return []
                entries = [entry] if entry is not None else []

        pending: list[dict[str, Any]] = []
        for entry in entries:
            request = entry.channel.get_pending_request()
            if request is None:
                continue
            pending.append(self._serialize_pending_interaction(entry, request))
        return pending

    def get_pending_interaction(self, agent_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        interactions = self.get_pending_interactions(agent_id=agent_id)
        if not interactions:
            return None
        return interactions[0]

    def respond_to_interaction(self, agent_id: str, response: Any) -> str:
        with self._lock:
            self._cleanup_finished_locked()
            try:
                entry = self._lookup_entry_locked(agent_id, include_completed=False)
            except ValueError as exc:
                return f"{exc}\n"
            if entry is None:
                return f"Unknown running agent id: {agent_id}\n"

        pending = entry.channel.get_pending_request()
        if pending is None:
            return f"No pending interaction request for agent '{entry.agent_name}' ({entry.agent_id}).\n"

        parsed = self._normalize_interaction_response(response, pending)
        if isinstance(parsed, str):
            return parsed
        approved_tool: Optional[str] = None
        if parsed.decision == "approve_session":
            candidate = str(pending.tool_name or "").strip()
            if candidate:
                approved_tool = candidate
        try:
            entry.channel.post_response(parsed)
        except Exception as exc:
            return f"Failed to post interaction response: {exc}\n"
        if approved_tool:
            with self._lock:
                self._session_tool_approvals.add(approved_tool)
        self._emit_ui_event(
            AgentRuntimeUIEvent(
                event_type="interaction_response",
                agent_id=entry.agent_id,
                agent_name=entry.agent_name,
                topic=entry.topic,
                request_id=str(pending.request_id or "").strip(),
                request_type=str(pending.request_type or "").strip(),
                decision=str(parsed.decision or "").strip(),
            )
        )
        return (
            f"Posted interaction response to agent '{entry.agent_name}' "
            f"({entry.agent_id}).\n"
        )

    def shutdown(self, *, join_timeout_s: Optional[float] = None) -> None:
        timeout = self._join_timeout_s if join_timeout_s is None else max(0.1, float(join_timeout_s))
        with self._lock:
            entries = list(self._agents.values())
            scheduler = self._scheduler
            self._scheduler = None
        if scheduler is not None:
            try:
                scheduler.stop()
            except Exception:
                logger.exception("Failed to stop scheduler during runtime manager shutdown")

        for entry in entries:
            self._cancel_entry_pending_interactions(
                entry,
                reason="Application shutting down",
                close_channel=True,
            )
            self._safe_call(entry.harness, "stop")
            self._safe_call(entry.agent, "stop")

        for entry in entries:
            if entry.thread.is_alive():
                entry.thread.join(timeout=timeout)

        with self._lock:
            self._cleanup_finished_locked()
            remaining = list(self._agents.values())
        if remaining:
            ids = ", ".join(entry.agent_id for entry in remaining)
            logger.warning(
                "Runtime manager shutdown completed with %d still-running agents: %s",
                len(remaining),
                ids,
            )

    def _mark_run_completed(self, agent_id: str) -> None:
        event: Optional[AgentRuntimeUIEvent] = None
        with self._lock:
            entry = self._agents.get(agent_id)
            if entry is None:
                return
            event = AgentRuntimeUIEvent(
                event_type="run_completed",
                agent_id=entry.agent_id,
                agent_name=entry.agent_name,
                topic=entry.topic,
                status=self._entry_status_value(entry),
            )
            self._finalize_entry_locked(entry)
        if event is not None:
            self._emit_ui_event(event)

    def _resolve_max_concurrent(self, app: Any) -> int:
        agent_cfg = getattr(getattr(app, "config", None), "agents", None)
        if agent_cfg is None:
            return 5
        try:
            return max(1, int(getattr(agent_cfg, "max_concurrent", 5)))
        except Exception:
            return 5

    def _resolve_log_stream_queue_limit(self, app: Any) -> int:
        agent_cfg = getattr(getattr(app, "config", None), "agents", None)
        if agent_cfg is None:
            return self._DEFAULT_LOG_STREAM_QUEUE_LIMIT
        try:
            return max(
                1,
                int(
                    getattr(
                        agent_cfg,
                        "log_stream_queue_limit",
                        self._DEFAULT_LOG_STREAM_QUEUE_LIMIT,
                    )
                ),
            )
        except Exception:
            return self._DEFAULT_LOG_STREAM_QUEUE_LIMIT

    def _attach_log_stream_to_agent(
        self,
        *,
        agent_id: str,
        agent: Any,
    ) -> None:
        state = getattr(agent, "state", None)
        if state is None:
            return
        add_log = getattr(state, "add_log", None)
        if not callable(add_log):
            return
        if bool(getattr(state, "_lsm_log_stream_wrapped", False)):
            return

        original_add_log = add_log

        def _wrapped_add_log(entry: AgentLogEntry) -> None:
            original_add_log(entry)
            self._enqueue_log_stream_entry(agent_id, entry)

        try:
            setattr(state, "add_log", _wrapped_add_log)
            setattr(state, "_lsm_log_stream_wrapped", True)
        except Exception:
            logger.exception(
                "Failed to attach log stream wrapper to agent '%s' (%s)",
                getattr(agent, "name", "agent"),
                agent_id,
            )

    def _serialize_stream_log_entry(self, entry: AgentLogEntry) -> dict[str, Any]:
        action_arguments = entry.action_arguments
        if isinstance(action_arguments, dict):
            normalized_args: Any = dict(action_arguments)
        else:
            normalized_args = action_arguments
        return {
            "timestamp": entry.timestamp.isoformat(),
            "actor": str(entry.actor or "").strip().lower(),
            "provider_name": str(entry.provider_name or "").strip(),
            "model_name": str(entry.model_name or "").strip(),
            "content": str(entry.content or ""),
            "action": str(entry.action or "").strip(),
            "action_arguments": normalized_args,
        }

    def _enqueue_log_stream_entry(self, agent_id: str, entry: AgentLogEntry) -> None:
        serialized = self._serialize_stream_log_entry(entry)
        with self._lock:
            stream = self._log_streams.get(agent_id)
            if stream is None:
                stream = _AgentLogStream(
                    max_entries=self._DEFAULT_LOG_STREAM_QUEUE_LIMIT,
                )
                self._log_streams[agent_id] = stream
            while len(stream.entries) >= max(1, int(stream.max_entries)):
                stream.entries.popleft()
                stream.dropped_count += 1
            stream.entries.append(serialized)

    def _emit_ui_event(self, event: AgentRuntimeUIEvent) -> None:
        sink: Optional[Callable[[AgentRuntimeUIEvent], None]]
        with self._lock:
            sink = self._ui_event_sink
        if not callable(sink):
            return
        try:
            sink(event)
        except Exception:
            logger.exception(
                "Failed to emit runtime UI event '%s' for agent '%s'.",
                event.event_type,
                event.agent_id,
            )

    def _build_harness(
        self,
        *,
        app: Any,
        sandbox: Any,
        agent_cfg: Any,
        agent_name: str,
        tool_registry: Any,
        agent: Any,
        memory_store: Any,
        interaction_channel: InteractionChannel,
        log_callback: Optional[Callable[[AgentLogEntry], None]] = None,
    ) -> Any:
        try:
            effective_agent_cfg = getattr(agent, "agent_config", agent_cfg)
            tool_allowlist = getattr(agent, "tool_allowlist", None)
            harness_kwargs: dict[str, Any] = {
                "agent_name": agent_name,
                "tool_allowlist": tool_allowlist,
                "vectordb_config": getattr(app.config, "vectordb", None),
                "memory_store": memory_store,
                "interaction_channel": interaction_channel,
            }
            if log_callback is not None:
                harness_kwargs["log_callback"] = log_callback
            return AgentHarness(
                effective_agent_cfg,
                tool_registry,
                app.config.llm,
                sandbox,
                **harness_kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize AgentHarness for run '%s': %s",
                agent_name,
                exc,
            )
            return _NoopHarness()

    def _attach_interaction_channel_to_sandbox(
        self,
        *,
        sandbox: Any,
        channel: InteractionChannel,
        topic: str,
    ) -> None:
        setter = getattr(sandbox, "set_interaction_channel", None)
        if not callable(setter):
            return

        def _waiting_callback(waiting: bool) -> None:
            _ = topic
            return

        try:
            setter(channel, waiting_state_callback=_waiting_callback)
        except Exception:
            logger.exception("Failed to set interaction channel on sandbox")

    def _empty_meta_snapshot(self) -> dict[str, Any]:
        return {
            "available": False,
            "status": "idle",
            "thread_alive": False,
            "goal": "",
            "execution_order": [],
            "tasks": [],
            "task_runs": [],
            "artifacts": [],
            "final_result_path": None,
            "meta_log_path": None,
        }

    def _latest_entry_for_agent_name_locked(self, agent_name: str) -> Optional[AgentRunEntry]:
        normalized = str(agent_name or "").strip().lower()
        active = [
            entry
            for entry in self._agents.values()
            if str(entry.agent_name).strip().lower() == normalized
        ]
        if active:
            return max(active, key=lambda item: item.started_at)

        completed = [
            self._completed_runs[item_id]
            for item_id in self._completed_order
            if item_id in self._completed_runs
            and str(self._completed_runs[item_id].agent_name).strip().lower() == normalized
        ]
        if completed:
            return completed[-1]
        return None

    def _cleanup_finished_locked(self) -> None:
        finished_ids = [
            entry_id
            for entry_id, entry in self._agents.items()
            if not entry.thread.is_alive()
        ]
        for entry_id in finished_ids:
            entry = self._agents.get(entry_id)
            if entry is not None:
                self._finalize_entry_locked(entry)

    def _finalize_entry_locked(self, entry: AgentRunEntry) -> None:
        active_entry = self._agents.pop(entry.agent_id, None)
        target = active_entry or entry
        target.completed_at = datetime.utcnow()
        try:
            target.channel.shutdown("Agent run completed")
        except Exception:
            logger.exception(
                "Failed to shutdown interaction channel for completed run '%s'",
                target.agent_id,
            )
        self._completed_runs[target.agent_id] = target
        if target.agent_id not in self._completed_order:
            self._completed_order.append(target.agent_id)
        if self._selected_agent_id == target.agent_id:
            self._selected_agent_id = next(iter(self._agents.keys()), None)
        self._prune_completed_locked()

    def _prune_completed_locked(self) -> None:
        while len(self._completed_order) > self._completed_retention:
            oldest_id = self._completed_order.pop(0)
            oldest = self._completed_runs.pop(oldest_id, None)
            self._log_streams.pop(oldest_id, None)
            if oldest is None:
                continue
            try:
                oldest.channel.shutdown("Completed run evicted from history")
            except Exception:
                logger.exception(
                    "Failed to shutdown interaction channel for pruned run '%s'",
                    oldest_id,
                )

    def _lookup_entry_locked(
        self,
        agent_id: str,
        *,
        include_completed: bool,
    ) -> Optional[AgentRunEntry]:
        normalized = str(agent_id or "").strip()
        if not normalized:
            return None
        if normalized in self._agents:
            return self._agents[normalized]
        if include_completed and normalized in self._completed_runs:
            return self._completed_runs[normalized]

        candidates = list(self._agents.keys())
        if include_completed:
            candidates.extend(self._completed_runs.keys())
        prefix_matches = [item for item in candidates if item.startswith(normalized)]
        if len(prefix_matches) == 1:
            match = prefix_matches[0]
            if match in self._agents:
                return self._agents[match]
            return self._completed_runs.get(match)
        if len(prefix_matches) > 1:
            raise ValueError(
                f"Ambiguous agent id '{normalized}'. Matches: {', '.join(prefix_matches[:5])}"
            )
        return None

    def _resolve_target_entry_locked(
        self,
        *,
        agent_id: Optional[str],
        action: str,
        include_completed: bool,
    ) -> tuple[Optional[AgentRunEntry], Optional[str]]:
        if agent_id is not None:
            try:
                entry = self._lookup_entry_locked(
                    agent_id,
                    include_completed=include_completed,
                )
            except ValueError as exc:
                return None, f"{exc}\n"
            if entry is None:
                return None, f"Unknown agent id: {agent_id}\n"
            self._selected_agent_id = entry.agent_id
            return entry, None

        selected_id = str(self._selected_agent_id or "").strip()
        if selected_id:
            selected_entry = self._agents.get(selected_id)
            if selected_entry is not None:
                return selected_entry, None
            if include_completed:
                completed_selected_entry = self._completed_runs.get(selected_id)
                if completed_selected_entry is not None:
                    return completed_selected_entry, None

        if len(self._agents) == 1:
            entry = next(iter(self._agents.values()))
            self._selected_agent_id = entry.agent_id
            return entry, None
        if len(self._agents) > 1:
            return (
                None,
                f"Multiple active agents. Specify an agent_id for /agent {action}.\n",
            )
        if include_completed:
            if len(self._completed_runs) == 1:
                return next(iter(self._completed_runs.values())), None
            if len(self._completed_runs) > 1:
                return (
                    None,
                    f"Multiple recent agents found. Specify an agent_id for /agent {action}.\n",
                )
        return None, "No active agent.\n"

    def _entry_status_value(self, entry: AgentRunEntry) -> str:
        return str(getattr(entry.agent.state.status, "value", entry.agent.state.status))

    def _format_entry_status(self, entry: AgentRunEntry) -> str:
        status = self._entry_status_value(entry)
        alive = bool(entry.thread.is_alive())
        logs = len(getattr(entry.agent.state, "log_entries", []))
        completed_at = (
            entry.completed_at.isoformat()
            if entry.completed_at is not None
            else "-"
        )
        return (
            f"Agent ID: {entry.agent_id}\n"
            f"Agent: {entry.agent_name}\n"
            f"Topic: {entry.topic}\n"
            f"Status: {status}\n"
            f"Thread alive: {alive}\n"
            f"Started at: {entry.started_at.isoformat()}\n"
            f"Completed at: {completed_at}\n"
            f"Logs: {logs} entries\n"
        )

    def _format_compact_entry_status(
        self,
        entry: AgentRunEntry,
        *,
        include_topic: bool = False,
    ) -> str:
        status = self._entry_status_value(entry)
        age_seconds = max(0.0, (datetime.utcnow() - entry.started_at).total_seconds())
        topic_part = f" | topic={entry.topic}" if include_topic else ""
        return (
            f"- {entry.agent_id[:8]} | {entry.agent_name} | status={status} "
            f"| alive={entry.thread.is_alive()} | age={round(age_seconds, 1)}s{topic_part}"
        )

    def _cancel_entry_pending_interactions(
        self,
        entry: AgentRunEntry,
        *,
        reason: str,
        close_channel: bool = False,
    ) -> None:
        try:
            if close_channel:
                entry.channel.shutdown(reason)
            else:
                entry.channel.cancel_pending(reason)
        except Exception:
            logger.exception(
                "Failed to cancel pending interaction for agent '%s' (%s)",
                entry.agent_name,
                entry.agent_id,
            )

    @staticmethod
    def _wait_for_thread_completion(
        thread: threading.Thread,
        *,
        timeout_s: float,
    ) -> bool:
        timeout = max(0.1, float(timeout_s))
        deadline = time.monotonic() + timeout
        while thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            thread.join(timeout=min(0.2, remaining))
        return True

    def _serialize_pending_interaction(
        self,
        entry: AgentRunEntry,
        request: InteractionRequest,
    ) -> dict[str, Any]:
        return {
            "agent_id": entry.agent_id,
            "agent_name": entry.agent_name,
            "topic": entry.topic,
            "request_id": request.request_id,
            "request_type": request.request_type,
            "tool_name": request.tool_name,
            "risk_level": request.risk_level,
            "reason": request.reason,
            "args_summary": request.args_summary,
            "prompt": request.prompt,
            "timestamp": request.timestamp.isoformat(),
        }

    def _normalize_interaction_response(
        self,
        response: Any,
        pending: InteractionRequest,
    ) -> InteractionResponse | str:
        if isinstance(response, InteractionResponse):
            return response
        if not isinstance(response, dict):
            return "Interaction response must be an object.\n"
        request_id = str(response.get("request_id") or pending.request_id).strip()
        decision = str(response.get("decision", "")).strip()
        user_message = str(response.get("user_message", "")).strip()
        try:
            return InteractionResponse(
                request_id=request_id,
                decision=decision,
                user_message=user_message,
            )
        except ValueError as exc:
            return f"Invalid interaction response: {exc}\n"

    def _safe_call(self, target: Any, method_name: str) -> None:
        method = getattr(target, method_name, None)
        if not callable(method):
            return
        try:
            method()
        except Exception:
            logger.exception("Failed to call %s on %r", method_name, target)

    def list_schedules(self, app: Any) -> list[dict[str, Any]]:
        scheduler = self._ensure_scheduler(app, start=False)
        return scheduler.list_schedules()

    def format_schedules(self, app: Any) -> str:
        try:
            schedules = self.list_schedules(app)
        except Exception as exc:
            return f"{exc}\n"

        if not schedules:
            return "No schedules configured.\n"

        lines = [f"Schedules: {len(schedules)}"]
        for row in schedules:
            lines.append(
                f"- {row['id']} | enabled={row['enabled']} | agent={row['agent_name']} "
                f"| interval={row['interval']} | policy={row['concurrency_policy']} "
                f"| confirmation={row['confirmation_mode']}"
            )
        lines.append("")
        return "\n".join(lines)

    def format_schedule_status(self, app: Any) -> str:
        try:
            scheduler = self._ensure_scheduler(app, start=True)
            rows = scheduler.list_schedules()
        except Exception as exc:
            return f"{exc}\n"

        if not rows:
            return "No schedules configured.\n"

        lines = [f"Scheduler status ({len(rows)} schedule(s)):"]
        for row in rows:
            lines.append(
                f"- {row['id']} | status={row['last_status']} | running={row['running']} "
                f"| queued={row['queued_runs']} | next={row['next_run_at']} | last={row['last_run_at'] or '-'}"
            )
            if row.get("last_error"):
                lines.append(f"  error={row['last_error']}")
        lines.append("")
        return "\n".join(lines)

    def add_schedule(
        self,
        app: Any,
        *,
        agent_name: str,
        interval: str,
        params: Optional[dict[str, Any]] = None,
        concurrency_policy: str = "skip",
        confirmation_mode: str = "auto",
    ) -> str:
        normalized_params = dict(params or {})
        schedule = ScheduleConfig(
            agent_name=str(agent_name).strip(),
            params=normalized_params,
            interval=str(interval).strip(),
            enabled=True,
            concurrency_policy=str(concurrency_policy).strip(),
            confirmation_mode=str(confirmation_mode).strip(),
        )
        schedule.validate()

        agent_cfg = self._require_agent_config(app)
        agent_cfg.schedules.append(schedule)
        scheduler = self._rebuild_scheduler(app, start=True)
        self._persist_config_if_possible(app)
        snapshots = scheduler.list_schedules()
        schedule_id = snapshots[-1]["id"] if snapshots else f"{len(agent_cfg.schedules)-1}:{schedule.agent_name}"
        return (
            f"Added schedule '{schedule_id}' for agent '{schedule.agent_name}' "
            f"(interval={schedule.interval}, concurrency_policy={schedule.concurrency_policy}, "
            f"confirmation_mode={schedule.confirmation_mode}).\n"
        )

    def set_schedule_enabled(
        self,
        app: Any,
        schedule_id: str,
        *,
        enabled: bool,
    ) -> str:
        scheduler = self._ensure_scheduler(app, start=False)
        snapshots = scheduler.list_schedules()
        index = self._resolve_schedule_index(schedule_id, snapshots)
        if index is None:
            return f"Schedule not found: {schedule_id}\n"

        agent_cfg = self._require_agent_config(app)
        if index >= len(agent_cfg.schedules):
            return f"Schedule not found: {schedule_id}\n"
        agent_cfg.schedules[index].enabled = bool(enabled)
        self._rebuild_scheduler(app, start=True)
        self._persist_config_if_possible(app)
        action = "Enabled" if enabled else "Disabled"
        return f"{action} schedule '{snapshots[index]['id']}'.\n"

    def remove_schedule(self, app: Any, schedule_id: str) -> str:
        scheduler = self._ensure_scheduler(app, start=False)
        snapshots = scheduler.list_schedules()
        index = self._resolve_schedule_index(schedule_id, snapshots)
        if index is None:
            return f"Schedule not found: {schedule_id}\n"

        agent_cfg = self._require_agent_config(app)
        if index >= len(agent_cfg.schedules):
            return f"Schedule not found: {schedule_id}\n"
        removed = agent_cfg.schedules.pop(index)
        self._rebuild_scheduler(app, start=True)
        self._persist_config_if_possible(app)
        return (
            f"Removed schedule '{snapshots[index]['id']}' "
            f"(agent={removed.agent_name}, interval={removed.interval}).\n"
        )

    def handle_schedule_command(self, app: Any, command: str) -> str:
        try:
            parts = tokenize_command(command, use_shlex=True)
        except CommandParseError as exc:
            return format_command_error(f"Invalid command syntax: {exc}")

        if len(parts) < 3:
            return _schedule_help()

        action = parts[2].strip().lower()
        if action == "list":
            return self.format_schedules(app)
        if action == "status":
            return self.format_schedule_status(app)
        if action == "add":
            if len(parts) < 5:
                return (
                    "Usage: /agent schedule add <agent_name> <interval> "
                    "[--params '{\"topic\":\"...\"}'] "
                    "[--concurrency_policy skip|queue|cancel] "
                    "[--confirmation_mode auto|confirm|deny]\n"
                )
            agent_name = parts[3].strip()
            interval = parts[4].strip()
            params: dict[str, Any] = {}
            concurrency_policy = "skip"
            confirmation_mode = "auto"
            idx = 5
            while idx < len(parts):
                token = parts[idx].strip().lower()
                if token == "--params":
                    idx += 1
                    if idx >= len(parts):
                        return "Missing value for --params.\n"
                    try:
                        parsed = json.loads(parts[idx])
                    except json.JSONDecodeError as exc:
                        return f"Invalid --params JSON: {exc}\n"
                    if not isinstance(parsed, dict):
                        return "--params must decode to a JSON object.\n"
                    params = parsed
                    idx += 1
                    continue
                if token in {"--concurrency_policy", "--concurrency-policy"}:
                    idx += 1
                    if idx >= len(parts):
                        return "Missing value for --concurrency_policy.\n"
                    concurrency_policy = parts[idx].strip().lower()
                    idx += 1
                    continue
                if token in {"--confirmation_mode", "--confirmation-mode"}:
                    idx += 1
                    if idx >= len(parts):
                        return "Missing value for --confirmation_mode.\n"
                    confirmation_mode = parts[idx].strip().lower()
                    idx += 1
                    continue
                return (
                    "Usage: /agent schedule add <agent_name> <interval> "
                    "[--params '{\"topic\":\"...\"}'] "
                    "[--concurrency_policy skip|queue|cancel] "
                    "[--confirmation_mode auto|confirm|deny]\n"
                )
            try:
                return self.add_schedule(
                    app,
                    agent_name=agent_name,
                    interval=interval,
                    params=params,
                    concurrency_policy=concurrency_policy,
                    confirmation_mode=confirmation_mode,
                )
            except Exception as exc:
                return f"Failed to add schedule: {exc}\n"
        if action in {"enable", "disable"}:
            if len(parts) < 4:
                return f"Usage: /agent schedule {action} <schedule_id>\n"
            try:
                return self.set_schedule_enabled(
                    app,
                    parts[3].strip(),
                    enabled=action == "enable",
                )
            except Exception as exc:
                return f"Failed to {action} schedule: {exc}\n"
        if action == "remove":
            if len(parts) < 4:
                return "Usage: /agent schedule remove <schedule_id>\n"
            try:
                return self.remove_schedule(app, parts[3].strip())
            except Exception as exc:
                return f"Failed to remove schedule: {exc}\n"
        return _schedule_help()

    def _ensure_scheduler(self, app: Any, *, start: bool) -> AgentScheduler:
        with self._lock:
            scheduler = self._scheduler
            if scheduler is None:
                scheduler = self._build_scheduler(app)
                self._scheduler = scheduler
            if start and self._has_enabled_schedules(app):
                scheduler.start()
            return scheduler

    def _rebuild_scheduler(self, app: Any, *, start: bool) -> AgentScheduler:
        with self._lock:
            existing = self._scheduler
            if existing is not None:
                existing.stop()
            scheduler = self._build_scheduler(app)
            self._scheduler = scheduler
            if start and self._has_enabled_schedules(app):
                scheduler.start()
            return scheduler

    @staticmethod
    def _has_enabled_schedules(app: Any) -> bool:
        agent_cfg = getattr(getattr(app, "config", None), "agents", None)
        if agent_cfg is None:
            return False
        schedules = getattr(agent_cfg, "schedules", [])
        return any(getattr(schedule, "enabled", False) for schedule in schedules)

    @staticmethod
    def _require_agent_config(app: Any) -> Any:
        agent_cfg = getattr(getattr(app, "config", None), "agents", None)
        if agent_cfg is None or not getattr(agent_cfg, "enabled", False):
            raise RuntimeError("Agents are disabled. Enable `agents.enabled` in config.")
        return agent_cfg

    @staticmethod
    def _build_scheduler(app: Any) -> AgentScheduler:
        cfg = getattr(app, "config", None)
        if cfg is None:
            raise RuntimeError("Application config is required for scheduler commands.")
        return AgentScheduler(
            cfg,
            collection=getattr(app, "query_provider", None),
            embedder=getattr(app, "query_embedder", None),
            batch_size=getattr(cfg, "batch_size", 32),
        )

    @staticmethod
    def _persist_config_if_possible(app: Any) -> None:
        cfg = getattr(app, "config", None)
        if cfg is None:
            return
        config_path = getattr(cfg, "config_path", None)
        if config_path in {None, ""}:
            return
        try:
            save_config_to_file(cfg, config_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to persist config to '{config_path}': {exc}") from exc

    @staticmethod
    def _resolve_schedule_index(
        schedule_id: str,
        snapshots: list[dict[str, Any]],
    ) -> Optional[int]:
        normalized = str(schedule_id or "").strip()
        if not normalized:
            return None

        for row in snapshots:
            if str(row.get("id", "")) == normalized:
                return int(row.get("index", -1))

        if normalized.isdigit():
            idx = int(normalized)
            if any(int(row.get("index", -1)) == idx for row in snapshots):
                return idx
            return None

        if ":" in normalized:
            prefix = normalized.split(":", 1)[0]
            if prefix.isdigit():
                idx = int(prefix)
                if any(int(row.get("index", -1)) == idx for row in snapshots):
                    return idx
        return None

    def get_memory_candidates(
        self,
        app: Any,
        *,
        status: Optional[str] = "pending",
        limit: int = 100,
    ) -> list[MemoryCandidate]:
        store = self._open_memory_store(app)
        try:
            normalized_status = None if status in {None, "", "all"} else str(status).strip().lower()
            return store.list_candidates(status=normalized_status, limit=max(1, int(limit)))
        finally:
            store.close()

    def format_memory_candidates(
        self,
        app: Any,
        *,
        status: Optional[str] = "pending",
        limit: int = 25,
    ) -> str:
        try:
            candidates = self.get_memory_candidates(app, status=status, limit=limit)
        except Exception as exc:
            return f"{exc}\n"
        if not candidates:
            status_label = str(status or "all")
            return f"No memory candidates found ({status_label}).\n"

        status_label = str(status or "all")
        lines = [f"Memory candidates ({status_label}): {len(candidates)}"]
        for candidate in candidates:
            memory = candidate.memory
            lines.append(
                f"- {candidate.id} | key={memory.key} | type={memory.type} | scope={memory.scope}"
            )
            lines.append(
                f"  status={candidate.status} confidence={memory.confidence:.2f} "
                f"tags={','.join(memory.tags) if memory.tags else '-'}"
            )
            lines.append(f"  rationale={candidate.rationale}")
        lines.append("")
        return "\n".join(lines)

    def promote_memory_candidate(self, app: Any, candidate_id: str) -> str:
        normalized_id = str(candidate_id).strip()
        if not normalized_id:
            return "Usage: /memory promote <candidate_id>\n"
        store = self._open_memory_store(app)
        try:
            memory = store.promote(normalized_id)
            return (
                f"Promoted memory candidate '{normalized_id}' "
                f"as memory '{memory.key}' ({memory.id}).\n"
            )
        except KeyError:
            return f"Memory candidate not found: {normalized_id}\n"
        except Exception as exc:
            return f"Failed to promote memory candidate '{normalized_id}': {exc}\n"
        finally:
            store.close()

    def reject_memory_candidate(self, app: Any, candidate_id: str) -> str:
        normalized_id = str(candidate_id).strip()
        if not normalized_id:
            return "Usage: /memory reject <candidate_id>\n"
        store = self._open_memory_store(app)
        try:
            store.reject(normalized_id)
            return f"Rejected memory candidate '{normalized_id}'.\n"
        except KeyError:
            return f"Memory candidate not found: {normalized_id}\n"
        except Exception as exc:
            return f"Failed to reject memory candidate '{normalized_id}': {exc}\n"
        finally:
            store.close()

    def edit_memory_candidate_ttl(
        self,
        app: Any,
        candidate_id: str,
        ttl_days: int,
    ) -> str:
        normalized_id = str(candidate_id).strip()
        if not normalized_id:
            return "Usage: /memory ttl <candidate_id> <days>\n"
        if int(ttl_days) < 1:
            return "TTL days must be >= 1.\n"

        store = self._open_memory_store(app)
        try:
            candidate = self._find_candidate(store, normalized_id)
            if candidate is None:
                return f"Memory candidate not found: {normalized_id}\n"

            new_expiry = now_utc() + timedelta(days=int(ttl_days))
            updated_memory = Memory(
                id=candidate.memory.id,
                type=candidate.memory.type,
                key=candidate.memory.key,
                value=candidate.memory.value,
                scope=candidate.memory.scope,
                tags=list(candidate.memory.tags),
                confidence=float(candidate.memory.confidence),
                created_at=candidate.memory.created_at,
                last_used_at=candidate.memory.last_used_at,
                expires_at=new_expiry,
                source_run_id=candidate.memory.source_run_id,
            )
            updated_memory.validate()
            replacement_id = self._replace_candidate_memory(
                store,
                existing_candidate=candidate,
                updated_memory=updated_memory,
                rationale=f"TTL updated to {int(ttl_days)} day(s) from memory UI.",
            )
            return (
                f"Updated TTL for memory candidate '{normalized_id}' "
                f"to {int(ttl_days)} day(s). Replacement candidate: {replacement_id}\n"
            )
        except Exception as exc:
            return f"Failed to update memory candidate TTL for '{normalized_id}': {exc}\n"
        finally:
            store.close()

    def _open_memory_store(self, app: Any):
        agent_cfg = getattr(app.config, "agents", None)
        if agent_cfg is None or not getattr(agent_cfg, "enabled", False):
            raise RuntimeError("Agents are disabled. Enable `agents.enabled` in config.")

        memory_cfg = getattr(agent_cfg, "memory", None)
        if memory_cfg is None or not getattr(memory_cfg, "enabled", False):
            raise RuntimeError("Agent memory is disabled. Enable `agents.memory.enabled` in config.")

        vectordb_cfg = getattr(app.config, "vectordb", None)
        if vectordb_cfg is None:
            raise RuntimeError("Vector DB config is required for memory commands.")
        return create_memory_store(agent_cfg, vectordb_cfg)

    @staticmethod
    def _find_candidate(store, candidate_id: str) -> Optional[MemoryCandidate]:
        for candidate in store.list_candidates(status=None, limit=10000):
            if candidate.id == candidate_id:
                return candidate
        return None

    @staticmethod
    def _replace_candidate_memory(
        store,
        *,
        existing_candidate: MemoryCandidate,
        updated_memory: Memory,
        rationale: str,
    ) -> str:
        existing_memory = existing_candidate.memory
        existing_status = existing_candidate.status
        store.delete(existing_memory.id)
        try:
            replacement_candidate_id = store.put_candidate(
                updated_memory,
                provenance="memory_ui_ttl_edit",
                rationale=rationale,
            )
            if existing_status == "promoted":
                store.promote(replacement_candidate_id)
            elif existing_status == "rejected":
                store.reject(replacement_candidate_id)
            return replacement_candidate_id
        except Exception:
            restore_candidate_id = store.put_candidate(
                existing_memory,
                provenance="memory_ui_ttl_restore",
                rationale="Automatic restore after failed TTL update.",
            )
            if existing_status == "promoted":
                store.promote(restore_candidate_id)
            elif existing_status == "rejected":
                store.reject(restore_candidate_id)
            raise


_MANAGER = AgentRuntimeManager()


def get_agent_runtime_manager() -> AgentRuntimeManager:
    """Return the singleton agent runtime manager."""
    return _MANAGER


def handle_memory_command(command: str, app: Any) -> str:
    """
    Parse and execute `/memory ...` commands.

    Supported:
    - `/memory candidates [pending|promoted|rejected|all]`
    - `/memory promote <candidate_id>`
    - `/memory reject <candidate_id>`
    - `/memory ttl <candidate_id> <days>`
    """
    text = command.strip()
    parts = tokenize_command(text)
    if len(parts) < 2:
        return _memory_help()

    manager = get_agent_runtime_manager()
    action = parts[1].lower()
    if action in {"candidates", "list"}:
        status = parts[2].strip().lower() if len(parts) > 2 else "pending"
        if status not in {"pending", "promoted", "rejected", "all"}:
            return "Usage: /memory candidates [pending|promoted|rejected|all]\n"
        return manager.format_memory_candidates(
            app,
            status=status,
            limit=50,
        )
    if action == "promote":
        if len(parts) < 3:
            return "Usage: /memory promote <candidate_id>\n"
        return manager.promote_memory_candidate(app, parts[2])
    if action == "reject":
        if len(parts) < 3:
            return "Usage: /memory reject <candidate_id>\n"
        return manager.reject_memory_candidate(app, parts[2])
    if action == "ttl":
        if len(parts) < 4:
            return "Usage: /memory ttl <candidate_id> <days>\n"
        candidate_id = parts[2].strip()
        try:
            ttl_days = int(parts[3].strip())
        except ValueError:
            return "TTL days must be an integer.\n"
        return manager.edit_memory_candidate_ttl(app, candidate_id, ttl_days)

    return _memory_help()


def handle_agent_command(command: str, app: Any) -> str:
    """
    Parse and execute `/agent ...` commands.

    Supported:
    - `/agent start <name> <topic...>`
    - `/agent queue [agent_id] <message>`
    - `/agent status`
    - `/agent stop`
    - `/agent pause`
    - `/agent resume [agent_id] [message]`
    - `/agent log`
    - `/agent meta start <goal...>`
    - `/agent meta status`
    - `/agent meta log`
    """
    text = command.strip()
    try:
        parts = tokenize_command(text, use_shlex=True)
    except CommandParseError as exc:
        return format_command_error(f"Invalid command syntax: {exc}")
    if len(parts) < 2:
        return _agent_help()

    manager = get_agent_runtime_manager()
    action = parts[1].lower()
    if action == "start":
        if len(parts) < 4:
            return "Usage: /agent start <name> <topic>\n"
        name = parts[2].strip()
        topic = " ".join(parts[3:]).strip()
        if not topic:
            return "Usage: /agent start <name> <topic>\n"
        return manager.start(app, name, topic)
    if action == "list":
        return manager.format_running_agents()
    if action == "interact":
        target_id = parts[2].strip() if len(parts) >= 3 else None
        return manager.format_pending_interactions(agent_id=target_id)
    if action in {"approve", "approve-session", "approve_session"}:
        if len(parts) < 3:
            usage = "/agent approve <agent_id>" if action == "approve" else "/agent approve-session <agent_id>"
            return f"Usage: {usage}\n"
        decision = "approve" if action == "approve" else "approve_session"
        return manager.respond_to_interaction(
            parts[2].strip(),
            {"decision": decision},
        )
    if action == "deny":
        if len(parts) < 3:
            return "Usage: /agent deny <agent_id> [reason]\n"
        payload: dict[str, Any] = {"decision": "deny"}
        reason = " ".join(parts[3:]).strip()
        if reason:
            payload["user_message"] = reason
        return manager.respond_to_interaction(parts[2].strip(), payload)
    if action == "reply":
        if len(parts) < 4:
            return "Usage: /agent reply <agent_id> <message>\n"
        message = " ".join(parts[3:]).strip()
        if not message:
            return "Usage: /agent reply <agent_id> <message>\n"
        return manager.respond_to_interaction(
            parts[2].strip(),
            {"decision": "reply", "user_message": message},
        )
    if action == "queue":
        parsed = _parse_agent_message_target_and_content(parts[2:])
        if parsed is None:
            return "Usage: /agent queue [agent_id] <message>\n"
        target_id, message = parsed
        return manager.queue_user_command(
            message,
            agent_id=target_id,
        )
    if action == "select":
        if len(parts) < 3:
            return "Usage: /agent select <agent_id>\n"
        return manager.select(parts[2].strip())
    if action == "status":
        target_id = parts[2].strip() if len(parts) >= 3 else None
        return manager.status(agent_id=target_id)
    if action == "meta":
        if len(parts) < 3:
            return _meta_help()
        subaction = parts[2].strip().lower()
        if subaction == "start":
            if len(parts) < 4:
                return "Usage: /agent meta start <goal>\n"
            goal = text.split(maxsplit=3)[3].strip()
            if not goal:
                return "Usage: /agent meta start <goal>\n"
            return manager.start_meta(app, goal)
        if subaction == "status":
            return manager.format_meta_status()
        if subaction == "log":
            return manager.meta_log()
        return _meta_help()
    if action == "schedule":
        return manager.handle_schedule_command(app, text)
    if action == "stop":
        target_id = parts[2].strip() if len(parts) >= 3 else None
        return manager.stop(agent_id=target_id)
    if action == "pause":
        target_id = parts[2].strip() if len(parts) >= 3 else None
        return manager.pause(agent_id=target_id)
    if action == "resume":
        parsed = _parse_resume_args(parts[2:])
        if parsed is None:
            return (
                "Usage: /agent resume [agent_id] [message]\n"
                "       /agent resume [--agent <agent_id>] [--message <message>]\n"
            )
        target_id, queued_message = parsed
        if queued_message:
            return manager.queue_user_command(
                queued_message,
                agent_id=target_id,
                resume_after_queue=True,
            )
        return manager.resume(agent_id=target_id)
    if action == "log":
        target_id = parts[2].strip() if len(parts) >= 3 else None
        return manager.log(agent_id=target_id)

    return _agent_help()


def _agent_help() -> str:
    return (
        "Agent commands:\n"
        "  /agent start <name> <topic>\n"
        "  /agent list\n"
        "  /agent interact [agent_id]\n"
        "  /agent approve <agent_id>\n"
        "  /agent deny <agent_id> [reason]\n"
        "  /agent approve-session <agent_id>\n"
        "  /agent reply <agent_id> <message>\n"
        "  /agent queue [agent_id] <message>\n"
        "  /agent select <agent_id>\n"
        "  /agent status [agent_id]\n"
        "  /agent stop [agent_id]\n"
        "  /agent pause [agent_id]\n"
        "  /agent resume [agent_id] [message]\n"
        "  /agent log [agent_id]\n"
        "  /agent meta start <goal>\n"
        "  /agent meta status\n"
        "  /agent meta log\n"
        "  /agent schedule add <agent_name> <interval> [--params '{\"topic\":\"...\"}'] "
        "[--concurrency_policy skip|queue|cancel] [--confirmation_mode auto|confirm|deny]\n"
        "  /agent schedule list\n"
        "  /agent schedule enable|disable <schedule_id>\n"
        "  /agent schedule remove <schedule_id>\n"
        "  /agent schedule status\n"
    )


def _meta_help() -> str:
    return (
        "Meta-agent commands:\n"
        "  /agent meta start <goal>\n"
        "  /agent meta status\n"
        "  /agent meta log\n"
    )


def _memory_help() -> str:
    return (
        "Memory commands:\n"
        "  /memory candidates [pending|promoted|rejected|all]\n"
        "  /memory promote <candidate_id>\n"
        "  /memory reject <candidate_id>\n"
        "  /memory ttl <candidate_id> <days>\n"
    )


def _schedule_help() -> str:
    return (
        "Agent schedule commands:\n"
        "  /agent schedule add <agent_name> <interval> [--params '{\"topic\":\"...\"}'] "
        "[--concurrency_policy skip|queue|cancel] [--confirmation_mode auto|confirm|deny]\n"
        "  /agent schedule list\n"
        "  /agent schedule enable <schedule_id>\n"
        "  /agent schedule disable <schedule_id>\n"
        "  /agent schedule remove <schedule_id>\n"
        "  /agent schedule status\n"
    )


_HEX_AGENT_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


def _looks_like_agent_id(value: str) -> bool:
    return bool(_HEX_AGENT_ID_RE.match(str(value or "").strip().lower()))


def _parse_agent_message_target_and_content(
    tokens: list[str],
) -> Optional[tuple[Optional[str], str]]:
    if not tokens:
        return None
    if tokens[0] in {"--agent", "-a"}:
        if len(tokens) < 3:
            return None
        target_id = str(tokens[1] or "").strip()
        message = " ".join(tokens[2:]).strip()
        if not target_id or not message:
            return None
        return target_id, message
    if len(tokens) >= 2 and _looks_like_agent_id(tokens[0]):
        target_id = str(tokens[0] or "").strip()
        message = " ".join(tokens[1:]).strip()
        if not message:
            return None
        return target_id, message
    message = " ".join(tokens).strip()
    if not message:
        return None
    return None, message


def _parse_resume_args(tokens: list[str]) -> Optional[tuple[Optional[str], str]]:
    if not tokens:
        return None, ""

    target_id: Optional[str] = None
    message_tokens: list[str] = []
    idx = 0

    if tokens[idx] in {"--agent", "-a"}:
        idx += 1
        if idx >= len(tokens):
            return None
        candidate = str(tokens[idx] or "").strip()
        if not candidate:
            return None
        target_id = candidate
        idx += 1
    elif _looks_like_agent_id(tokens[idx]):
        target_id = str(tokens[idx] or "").strip()
        idx += 1

    if idx < len(tokens) and tokens[idx] in {"--message", "-m"}:
        idx += 1
        if idx >= len(tokens):
            return None
        message_tokens = tokens[idx:]
    elif idx < len(tokens):
        message_tokens = tokens[idx:]

    message = " ".join(message_tokens).strip()
    return target_id, message
