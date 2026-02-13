"""
Shell-level agent command helpers.
"""

from __future__ import annotations

import json
import shlex
import threading
from datetime import datetime, timedelta
from typing import Any, Optional

from lsm.agents import AgentScheduler, create_agent
from lsm.agents.log_formatter import format_agent_log
from lsm.agents.memory import Memory, MemoryCandidate, create_memory_store
from lsm.agents.memory.models import now_utc
from lsm.agents.models import AgentContext
from lsm.agents.tools import ToolSandbox, create_default_tool_registry
from lsm.config.models import ScheduleConfig


class AgentRuntimeManager:
    """
    Manage active agent runtime instances for UI command surfaces.
    """

    def __init__(self) -> None:
        self._active_agent = None
        self._active_thread: Optional[threading.Thread] = None
        self._active_name: Optional[str] = None
        self._scheduler: Optional[AgentScheduler] = None
        self._lock = threading.Lock()

    def start(self, app: Any, agent_name: str, topic: str) -> str:
        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                return (
                    f"Agent '{self._active_name}' is already running. "
                    "Use /agent status or /agent stop first.\n"
                )

            agent_cfg = getattr(app.config, "agents", None)
            if agent_cfg is None or not getattr(agent_cfg, "enabled", False):
                return "Agents are disabled. Enable `agents.enabled` in config.\n"

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
            agent = create_agent(
                name=agent_name,
                llm_registry=app.config.llm,
                tool_registry=tool_registry,
                sandbox=sandbox,
                agent_config=agent_cfg,
            )
            context = AgentContext(
                messages=[{"role": "user", "content": topic}],
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
                finally:
                    if memory_store is not None:
                        memory_store.close()

            thread = threading.Thread(
                target=_run,
                daemon=True,
                name=f"Agent-{agent_name}",
            )
            self._active_agent = agent
            self._active_thread = thread
            self._active_name = agent_name
            thread.start()
            return f"Started agent '{agent_name}' with topic: {topic}\n"

    def status(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        alive = bool(self._active_thread and self._active_thread.is_alive())
        status = self._active_agent.state.status.value
        return (
            f"Agent: {self._active_name}\n"
            f"Status: {status}\n"
            f"Thread alive: {alive}\n"
            f"Logs: {len(self._active_agent.state.log_entries)} entries\n"
        )

    def stop(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.stop()
        return f"Stop requested for agent '{self._active_name}'.\n"

    def pause(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.pause()
        return f"Paused agent '{self._active_name}'.\n"

    def resume(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        self._active_agent.resume()
        return f"Resumed agent '{self._active_name}'.\n"

    def log(self) -> str:
        if self._active_agent is None:
            return "No active agent.\n"
        entries = self._active_agent.state.log_entries
        if not entries:
            return "No log entries yet.\n"
        return format_agent_log(entries)

    def get_active_agent(self):
        """Return the active agent instance for UI screens."""
        return self._active_agent

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
    ) -> str:
        normalized_params = dict(params or {})
        schedule = ScheduleConfig(
            agent_name=str(agent_name).strip(),
            params=normalized_params,
            interval=str(interval).strip(),
            enabled=True,
            concurrency_policy="skip",
            confirmation_mode="auto",
        )
        schedule.validate()

        agent_cfg = self._require_agent_config(app)
        agent_cfg.schedules.append(schedule)
        scheduler = self._rebuild_scheduler(app, start=True)
        snapshots = scheduler.list_schedules()
        schedule_id = snapshots[-1]["id"] if snapshots else f"{len(agent_cfg.schedules)-1}:{schedule.agent_name}"
        return (
            f"Added schedule '{schedule_id}' for agent '{schedule.agent_name}' "
            f"(interval={schedule.interval}).\n"
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
        return (
            f"Removed schedule '{snapshots[index]['id']}' "
            f"(agent={removed.agent_name}, interval={removed.interval}).\n"
        )

    def handle_schedule_command(self, app: Any, command: str) -> str:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return f"Invalid command syntax: {exc}\n"

        if len(parts) < 3:
            return _schedule_help()

        action = parts[2].strip().lower()
        if action == "list":
            return self.format_schedules(app)
        if action == "status":
            return self.format_schedule_status(app)
        if action == "add":
            if len(parts) < 5:
                return "Usage: /agent schedule add <agent_name> <interval> [--params '{\"topic\":\"...\"}']\n"
            agent_name = parts[3].strip()
            interval = parts[4].strip()
            params: dict[str, Any] = {}
            idx = 5
            while idx < len(parts):
                token = parts[idx].strip().lower()
                if token != "--params":
                    return (
                        "Usage: /agent schedule add <agent_name> <interval> "
                        "[--params '{\"topic\":\"...\"}']\n"
                    )
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
            try:
                return self.add_schedule(
                    app,
                    agent_name=agent_name,
                    interval=interval,
                    params=params,
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
    parts = text.split()
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
    - `/agent status`
    - `/agent stop`
    - `/agent pause`
    - `/agent resume`
    - `/agent log`
    """
    text = command.strip()
    parts = text.split()
    if len(parts) < 2:
        return _agent_help()

    manager = get_agent_runtime_manager()
    action = parts[1].lower()
    if action == "start":
        if len(parts) < 4:
            return "Usage: /agent start <name> <topic>\n"
        name = parts[2].strip()
        topic = text.split(maxsplit=3)[3].strip()
        if not topic:
            return "Usage: /agent start <name> <topic>\n"
        return manager.start(app, name, topic)
    if action == "status":
        return manager.status()
    if action == "schedule":
        return manager.handle_schedule_command(app, text)
    if action == "stop":
        return manager.stop()
    if action == "pause":
        return manager.pause()
    if action == "resume":
        return manager.resume()
    if action == "log":
        return manager.log()

    return _agent_help()


def _agent_help() -> str:
    return (
        "Agent commands:\n"
        "  /agent start <name> <topic>\n"
        "  /agent status\n"
        "  /agent stop\n"
        "  /agent pause\n"
        "  /agent resume\n"
        "  /agent log\n"
        "  /agent schedule add <agent_name> <interval> [--params '{\"topic\":\"...\"}']\n"
        "  /agent schedule list\n"
        "  /agent schedule enable|disable <schedule_id>\n"
        "  /agent schedule remove <schedule_id>\n"
        "  /agent schedule status\n"
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
        "  /agent schedule add <agent_name> <interval> [--params '{\"topic\":\"...\"}']\n"
        "  /agent schedule list\n"
        "  /agent schedule enable <schedule_id>\n"
        "  /agent schedule disable <schedule_id>\n"
        "  /agent schedule remove <schedule_id>\n"
        "  /agent schedule status\n"
    )
