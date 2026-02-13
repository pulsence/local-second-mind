"""
Agent scheduler service for recurring harness runs.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from lsm.config.models import LSMConfig, ScheduleConfig
from lsm.config.models.agents import AgentConfig, SandboxConfig
from lsm.logging import get_logger

from .base import AgentState, AgentStatus
from .factory import create_agent
from .harness import AgentHarness
from .memory import BaseMemoryStore, create_memory_store
from .models import AgentContext
from .tools import ToolSandbox, create_default_tool_registry

logger = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_schedule_id(index: int, schedule: ScheduleConfig) -> str:
    normalized = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_"
        for ch in str(schedule.agent_name or "").strip().lower()
    ).strip("_")
    if not normalized:
        normalized = "agent"
    return f"{index}:{normalized}"


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass
class _ScheduleRuntime:
    schedule_id: str
    index: int
    schedule: ScheduleConfig
    last_run_at: Optional[datetime]
    next_run_at: datetime
    last_status: str = "idle"
    last_error: Optional[str] = None
    queued_runs: int = 0
    running: bool = False
    harness: Optional[AgentHarness] = None
    thread: Optional[threading.Thread] = None


class AgentScheduler:
    """
    Persistent scheduler for recurring agent harness runs.
    """

    def __init__(
        self,
        config: LSMConfig,
        *,
        collection: Any = None,
        embedder: Any = None,
        batch_size: int = 32,
        tick_seconds: float = 60.0,
        now_fn: Optional[Callable[[], datetime]] = None,
        harness_cls: type[AgentHarness] = AgentHarness,
        agent_factory: Callable[..., Any] = create_agent,
        tool_registry_builder: Callable[..., Any] = create_default_tool_registry,
        memory_store_factory: Callable[[AgentConfig, Any], BaseMemoryStore] = create_memory_store,
    ) -> None:
        if config.agents is None or not config.agents.enabled:
            raise ValueError("Agent scheduler requires agents.enabled = true")

        self.config = config
        self.agent_config = config.agents
        self.collection = collection
        self.embedder = embedder
        self.batch_size = int(batch_size)
        self.tick_seconds = max(1.0, float(tick_seconds))
        self.now_fn = now_fn or _utcnow

        self.harness_cls = harness_cls
        self.agent_factory = agent_factory
        self.tool_registry_builder = tool_registry_builder
        self.memory_store_factory = memory_store_factory

        self.state_path = self.agent_config.agents_folder / "schedules.json"

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._entries: Dict[str, _ScheduleRuntime] = {}

        self._initialize_entries_from_config()
        self._save_state_locked()

    def start(self) -> threading.Thread:
        """
        Start the scheduler tick loop in a background thread.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="AgentScheduler",
            )
            self._thread.start()
            return self._thread

    def stop(self, *, join_timeout_s: float = 5.0) -> None:
        """
        Stop the scheduler loop and request graceful shutdown of active runs.
        """
        self._stop_event.set()
        with self._lock:
            entries = list(self._entries.values())
            thread = self._thread
        for entry in entries:
            harness = entry.harness
            if harness is not None:
                try:
                    harness.stop()
                except Exception:
                    logger.exception("Failed to stop harness for schedule '%s'", entry.schedule_id)
        if thread is not None:
            thread.join(timeout=max(0.1, float(join_timeout_s)))
        self.wait_until_idle(timeout_s=max(0.1, float(join_timeout_s)))
        with self._lock:
            self._save_state_locked()

    def tick(self, now: Optional[datetime] = None) -> None:
        """
        Process one scheduler tick.
        """
        current = now or self.now_fn()
        current = _parse_datetime(current) or self.now_fn()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        else:
            current = current.astimezone(timezone.utc)

        with self._lock:
            entry_ids = list(self._entries.keys())

        for schedule_id in entry_ids:
            self._process_entry(schedule_id, current)

        with self._lock:
            self._save_state_locked()

    def run_pending_once(self, now: Optional[datetime] = None) -> None:
        """
        Alias for one-shot tick invocation.
        """
        self.tick(now=now)

    def list_schedules(self) -> list[Dict[str, Any]]:
        """
        Return scheduler state snapshots.
        """
        with self._lock:
            snapshots: list[Dict[str, Any]] = []
            for entry in sorted(self._entries.values(), key=lambda item: item.index):
                snapshots.append(
                    {
                        "id": entry.schedule_id,
                        "index": entry.index,
                        "agent_name": entry.schedule.agent_name,
                        "interval": entry.schedule.interval,
                        "enabled": entry.schedule.enabled,
                        "concurrency_policy": entry.schedule.concurrency_policy,
                        "confirmation_mode": entry.schedule.confirmation_mode,
                        "params": dict(entry.schedule.params),
                        "last_run_at": entry.last_run_at.isoformat() if entry.last_run_at else None,
                        "next_run_at": entry.next_run_at.isoformat(),
                        "last_status": entry.last_status,
                        "last_error": entry.last_error,
                        "queued_runs": int(entry.queued_runs),
                        "running": bool(entry.running),
                    }
                )
            return snapshots

    def wait_until_idle(self, *, timeout_s: float = 5.0) -> bool:
        """
        Wait until no scheduled runs are active.
        """
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() <= deadline:
            with self._lock:
                running = any(entry.running for entry in self._entries.values())
            if not running:
                return True
            time.sleep(0.02)
        return False

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("Scheduler tick failed")
            self._stop_event.wait(self.tick_seconds)

    def _initialize_entries_from_config(self) -> None:
        persisted = self._load_state_file()
        persisted_entries = {}
        for row in persisted.get("schedules", []) if isinstance(persisted, dict) else []:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                continue
            persisted_entries[row_id] = row

        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        entries: Dict[str, _ScheduleRuntime] = {}
        for index, schedule in enumerate(self.agent_config.schedules):
            schedule_id = _safe_schedule_id(index, schedule)
            persisted_row = persisted_entries.get(schedule_id, {})

            last_run_at = _parse_datetime(persisted_row.get("last_run_at"))
            next_run_at = _parse_datetime(persisted_row.get("next_run_at"))
            if next_run_at is None:
                next_run_at = self._compute_next_run(schedule, now)

            entries[schedule_id] = _ScheduleRuntime(
                schedule_id=schedule_id,
                index=index,
                schedule=schedule,
                last_run_at=last_run_at,
                next_run_at=next_run_at,
                last_status=str(persisted_row.get("last_status", "idle")),
                last_error=(
                    str(persisted_row.get("last_error"))
                    if persisted_row.get("last_error") is not None
                    else None
                ),
                queued_runs=max(0, int(persisted_row.get("queued_runs", 0))),
            )

        with self._lock:
            self._entries = entries

    def _load_state_file(self) -> Dict[str, Any]:
        try:
            if not self.state_path.exists():
                return {}
            raw = self.state_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {}
            return parsed
        except Exception:
            logger.exception("Failed to load scheduler state from '%s'", self.state_path)
            return {}

    def _save_state_locked(self) -> None:
        payload = {
            "version": 1,
            "updated_at": self.now_fn().isoformat(),
            "schedules": [
                {
                    "id": entry.schedule_id,
                    "index": entry.index,
                    "agent_name": entry.schedule.agent_name,
                    "interval": entry.schedule.interval,
                    "enabled": entry.schedule.enabled,
                    "concurrency_policy": entry.schedule.concurrency_policy,
                    "confirmation_mode": entry.schedule.confirmation_mode,
                    "params": dict(entry.schedule.params),
                    "last_run_at": entry.last_run_at.isoformat() if entry.last_run_at else None,
                    "next_run_at": entry.next_run_at.isoformat(),
                    "last_status": entry.last_status,
                    "last_error": entry.last_error,
                    "queued_runs": int(entry.queued_runs),
                    "running": bool(entry.running),
                }
                for entry in sorted(self._entries.values(), key=lambda item: item.index)
            ],
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to persist scheduler state to '%s'", self.state_path)

    def _process_entry(self, schedule_id: str, now: datetime) -> None:
        with self._lock:
            entry = self._entries.get(schedule_id)
            if entry is None:
                return
            if not entry.schedule.enabled:
                return

            if entry.running:
                if now >= entry.next_run_at:
                    self._handle_overlap_locked(entry, now)
                return

            should_start = False
            if entry.queued_runs > 0:
                entry.queued_runs -= 1
                should_start = True
            elif now >= entry.next_run_at:
                should_start = True

        if should_start:
            self._start_schedule_run(schedule_id, now)

    def _handle_overlap_locked(self, entry: _ScheduleRuntime, now: datetime) -> None:
        policy = str(entry.schedule.concurrency_policy or "skip").strip().lower()
        if policy == "skip":
            entry.last_status = "skipped"
            entry.last_error = "Skipped overlapping run (concurrency_policy=skip)."
            entry.next_run_at = self._advance_next_run(entry.schedule, entry.next_run_at, now)
            return

        if policy == "queue":
            entry.queued_runs += 1
            entry.last_status = "queued"
            entry.last_error = None
            entry.next_run_at = self._advance_next_run(entry.schedule, entry.next_run_at, now)
            return

        # cancel
        harness = entry.harness
        if harness is not None:
            try:
                harness.stop()
            except Exception:
                logger.exception("Failed to request harness stop for schedule '%s'", entry.schedule_id)
        entry.queued_runs = max(1, entry.queued_runs)
        entry.last_status = "cancel_requested"
        entry.last_error = None
        entry.next_run_at = self._advance_next_run(entry.schedule, entry.next_run_at, now)

    def _start_schedule_run(self, schedule_id: str, now: datetime) -> None:
        with self._lock:
            entry = self._entries.get(schedule_id)
            if entry is None or entry.running or not entry.schedule.enabled:
                return
            schedule = entry.schedule

        try:
            run_bundle = self._build_run_bundle(schedule)
        except Exception as exc:
            with self._lock:
                entry = self._entries.get(schedule_id)
                if entry is None:
                    return
                entry.last_status = "failed"
                entry.last_error = f"Failed to initialize scheduled run: {exc}"
                entry.next_run_at = self._advance_next_run(
                    entry.schedule,
                    entry.next_run_at,
                    now,
                )
                self._save_state_locked()
            return

        harness = run_bundle["harness"]
        context = run_bundle["context"]
        memory_store = run_bundle.get("memory_store")

        with self._lock:
            entry = self._entries.get(schedule_id)
            if entry is None:
                if memory_store is not None:
                    memory_store.close()
                return
            entry.running = True
            entry.harness = harness
            entry.last_status = "running"
            entry.last_error = None
            entry.last_run_at = now
            entry.next_run_at = self._advance_next_run(
                entry.schedule,
                entry.next_run_at,
                now,
            )

            thread = threading.Thread(
                target=self._run_schedule_thread,
                args=(schedule_id, harness, context, memory_store),
                daemon=True,
                name=f"ScheduledRun-{schedule_id}",
            )
            entry.thread = thread
            self._save_state_locked()
            thread.start()

    def _run_schedule_thread(
        self,
        schedule_id: str,
        harness: AgentHarness,
        context: AgentContext,
        memory_store: Optional[BaseMemoryStore],
    ) -> None:
        state: Optional[AgentState] = None
        error: Optional[str] = None
        try:
            state = harness.run(context)
        except Exception as exc:
            error = str(exc)
            logger.exception("Scheduled harness run failed for '%s'", schedule_id)
        finally:
            if memory_store is not None:
                try:
                    memory_store.close()
                except Exception:
                    logger.exception("Failed to close memory store for '%s'", schedule_id)

        now = self.now_fn()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        trigger_follow_up_tick = False
        with self._lock:
            entry = self._entries.get(schedule_id)
            if entry is None:
                return
            entry.running = False
            entry.harness = None
            entry.thread = None

            if error is not None:
                entry.last_status = "failed"
                entry.last_error = error
            elif state is None:
                entry.last_status = "failed"
                entry.last_error = "Scheduled run returned no state"
            else:
                status_value = state.status.value if hasattr(state.status, "value") else str(state.status)
                entry.last_status = str(status_value)
                if entry.last_status == AgentStatus.FAILED.value:
                    entry.last_error = self._extract_run_error(state)
                else:
                    entry.last_error = None

            entry.next_run_at = self._advance_next_run(
                entry.schedule,
                entry.next_run_at,
                now,
            )
            if entry.queued_runs > 0 and not self._stop_event.is_set():
                entry.next_run_at = min(entry.next_run_at, now)
                trigger_follow_up_tick = True
            self._save_state_locked()

        if trigger_follow_up_tick and not self._stop_event.is_set():
            self.tick(now=now)

    @staticmethod
    def _extract_run_error(state: AgentState) -> Optional[str]:
        for entry in reversed(state.log_entries):
            text = str(entry.content or "").strip()
            if not text:
                continue
            if "failed" in text.lower() or "error" in text.lower():
                return text
        return None

    def _build_run_bundle(self, schedule: ScheduleConfig) -> Dict[str, Any]:
        permissions = self._resolve_schedule_permissions(schedule)

        memory_store: Optional[BaseMemoryStore] = None
        if self.agent_config.memory.enabled:
            memory_store = self.memory_store_factory(self.agent_config, self.config.vectordb)

        tool_registry = self.tool_registry_builder(
            self.config,
            collection=self.collection,
            embedder=self.embedder,
            batch_size=self.batch_size,
            memory_store=memory_store,
        )

        sandbox_config = self._build_sandbox_for_schedule(schedule, permissions)
        sandbox = ToolSandbox(sandbox_config)

        agent = self.agent_factory(
            name=schedule.agent_name,
            llm_registry=self.config.llm,
            tool_registry=tool_registry,
            sandbox=sandbox,
            agent_config=self.agent_config,
        )
        effective_agent_config = getattr(agent, "agent_config", self.agent_config)
        allowlist = self._build_schedule_allowlist(
            schedule=schedule,
            tool_definitions=tool_registry.list_definitions(),
            agent_allowlist=getattr(agent, "tool_allowlist", None),
            permissions=permissions,
        )

        harness = self.harness_cls(
            effective_agent_config,
            tool_registry,
            self.config.llm,
            sandbox,
            agent_name=schedule.agent_name,
            tool_allowlist=allowlist,
            vectordb_config=self.config.vectordb,
            memory_store=memory_store,
        )
        topic = str(schedule.params.get("topic", "")).strip()
        if not topic:
            topic = f"Scheduled run: {schedule.agent_name}"
        context = AgentContext(
            messages=[{"role": "user", "content": topic}],
            budget_tracking={
                "tokens_used": 0,
                "max_tokens_budget": int(effective_agent_config.max_tokens_budget),
                "started_at": self.now_fn().isoformat(),
                "scheduled": True,
            },
        )
        return {
            "harness": harness,
            "context": context,
            "memory_store": memory_store,
        }

    def _resolve_schedule_permissions(self, schedule: ScheduleConfig) -> Dict[str, bool]:
        params = schedule.params if isinstance(schedule.params, dict) else {}
        confirmation_mode = str(schedule.confirmation_mode or "auto").strip().lower()
        allow_writes = _to_bool(params.get("allow_writes"), default=False)
        allow_network = _to_bool(params.get("allow_network"), default=False)
        allow_exec = _to_bool(params.get("allow_exec"), default=False)
        force_docker = _to_bool(params.get("force_docker"), default=False)

        if confirmation_mode == "deny":
            allow_writes = False
            allow_network = False
            allow_exec = False

        effective_writes = allow_writes and bool(self.agent_config.sandbox.allowed_write_paths)
        effective_network = allow_network and bool(self.agent_config.sandbox.allow_url_access)
        effective_exec = allow_exec

        return {
            "allow_writes": effective_writes,
            "allow_network": effective_network,
            "allow_exec": effective_exec,
            "force_docker": force_docker,
        }

    def _build_sandbox_for_schedule(
        self,
        schedule: ScheduleConfig,
        permissions: Dict[str, bool],
    ) -> SandboxConfig:
        base = self.agent_config.sandbox
        allow_writes = bool(permissions.get("allow_writes", False))
        allow_network = bool(permissions.get("allow_network", False))
        allow_exec = bool(permissions.get("allow_exec", False))
        force_docker = bool(permissions.get("force_docker", False))
        confirmation_mode = str(schedule.confirmation_mode or "auto").strip().lower()

        require_user_permission: Dict[str, bool] = {}
        require_permission_by_risk: Dict[str, bool] = {}
        if confirmation_mode == "confirm":
            if allow_writes:
                require_permission_by_risk["writes_workspace"] = True
            if allow_network:
                require_permission_by_risk["network"] = True
            if allow_exec:
                require_permission_by_risk["exec"] = True
        elif confirmation_mode not in {"auto", "deny"}:
            require_user_permission = dict(base.require_user_permission)
            require_permission_by_risk = dict(base.require_permission_by_risk)

        execution_mode = "local_only"
        if allow_network or allow_exec or force_docker:
            execution_mode = "prefer_docker"

        return SandboxConfig(
            allowed_read_paths=list(base.allowed_read_paths),
            allowed_write_paths=list(base.allowed_write_paths) if allow_writes else [],
            allow_url_access=allow_network,
            require_user_permission=require_user_permission,
            require_permission_by_risk=require_permission_by_risk,
            execution_mode=execution_mode,
            force_docker=force_docker,
            limits=dict(base.limits),
            docker=dict(base.docker),
            tool_llm_assignments=dict(base.tool_llm_assignments),
        )

    @staticmethod
    def _build_schedule_allowlist(
        *,
        schedule: ScheduleConfig,
        tool_definitions: Sequence[Dict[str, Any]],
        agent_allowlist: Optional[set[str]],
        permissions: Dict[str, bool],
    ) -> set[str]:
        allowed_risks = {"read_only"}
        if permissions.get("allow_writes", False):
            allowed_risks.add("writes_workspace")
        if permissions.get("allow_network", False):
            allowed_risks.add("network")
        if permissions.get("allow_exec", False):
            allowed_risks.add("exec")

        allowlist = {
            str(definition.get("name", "")).strip()
            for definition in tool_definitions
            if str(definition.get("name", "")).strip()
            and str(definition.get("risk_level", "read_only")).strip() in allowed_risks
        }

        if agent_allowlist:
            normalized_agent_allowlist = {
                str(name).strip() for name in agent_allowlist if str(name).strip()
            }
            allowlist &= normalized_agent_allowlist

        explicit_allowlist = schedule.params.get("tool_allowlist")
        if isinstance(explicit_allowlist, list):
            requested = {
                str(name).strip() for name in explicit_allowlist if str(name).strip()
            }
            allowlist &= requested

        return allowlist

    def _advance_next_run(
        self,
        schedule: ScheduleConfig,
        current_next: datetime,
        reference_time: datetime,
    ) -> datetime:
        next_run = current_next
        while next_run <= reference_time:
            next_run = self._compute_next_run(schedule, next_run)
        return next_run

    def _compute_next_run(self, schedule: ScheduleConfig, from_time: datetime) -> datetime:
        interval = str(schedule.interval or "daily").strip().lower()
        if interval == "hourly":
            return from_time + timedelta(hours=1)
        if interval == "daily":
            return from_time + timedelta(days=1)
        if interval == "weekly":
            return from_time + timedelta(weeks=1)
        if interval.endswith("s") and interval[:-1].isdigit():
            seconds = max(1, int(interval[:-1]))
            return from_time + timedelta(seconds=seconds)
        return self._compute_next_cron_time(interval, from_time)

    def _compute_next_cron_time(self, expression: str, from_time: datetime) -> datetime:
        parts = [part for part in str(expression or "").split() if part]
        if len(parts) not in {5, 6}:
            raise ValueError(f"Unsupported cron expression: '{expression}'")

        has_seconds = len(parts) == 6
        if has_seconds:
            sec_field, min_field, hour_field, dom_field, mon_field, dow_field = parts
            second_values, second_any = self._parse_cron_field(sec_field, 0, 59, allow_seven=False)
        else:
            min_field, hour_field, dom_field, mon_field, dow_field = parts
            second_values, second_any = ({0}, True)

        minute_values, minute_any = self._parse_cron_field(min_field, 0, 59, allow_seven=False)
        hour_values, hour_any = self._parse_cron_field(hour_field, 0, 23, allow_seven=False)
        dom_values, dom_any = self._parse_cron_field(dom_field, 1, 31, allow_seven=False)
        month_values, month_any = self._parse_cron_field(mon_field, 1, 12, allow_seven=False)
        dow_values, dow_any = self._parse_cron_field(dow_field, 0, 6, allow_seven=True)

        base = from_time.astimezone(timezone.utc).replace(microsecond=0)
        step = timedelta(seconds=1 if has_seconds else 60)
        candidate = base + step
        if not has_seconds:
            candidate = candidate.replace(second=0)
        upper_bound = base + timedelta(days=366)

        while candidate <= upper_bound:
            if self._cron_candidate_matches(
                candidate,
                second_values=second_values,
                second_any=second_any,
                minute_values=minute_values,
                minute_any=minute_any,
                hour_values=hour_values,
                hour_any=hour_any,
                dom_values=dom_values,
                dom_any=dom_any,
                month_values=month_values,
                month_any=month_any,
                dow_values=dow_values,
                dow_any=dow_any,
            ):
                return candidate
            candidate += step
        raise ValueError(f"No matching cron run time found within one year for '{expression}'")

    @staticmethod
    def _cron_candidate_matches(
        candidate: datetime,
        *,
        second_values: set[int],
        second_any: bool,
        minute_values: set[int],
        minute_any: bool,
        hour_values: set[int],
        hour_any: bool,
        dom_values: set[int],
        dom_any: bool,
        month_values: set[int],
        month_any: bool,
        dow_values: set[int],
        dow_any: bool,
    ) -> bool:
        if not second_any and candidate.second not in second_values:
            return False
        if not minute_any and candidate.minute not in minute_values:
            return False
        if not hour_any and candidate.hour not in hour_values:
            return False
        if not month_any and candidate.month not in month_values:
            return False

        dom_matches = candidate.day in dom_values
        cron_dow = (candidate.weekday() + 1) % 7
        dow_matches = cron_dow in dow_values

        if dom_any and dow_any:
            return True
        if dom_any:
            return dow_matches
        if dow_any:
            return dom_matches
        return dom_matches or dow_matches

    @staticmethod
    def _parse_cron_field(
        field: str,
        min_value: int,
        max_value: int,
        *,
        allow_seven: bool,
    ) -> tuple[set[int], bool]:
        text = str(field or "").strip()
        if text == "*":
            return set(range(min_value, max_value + 1)), True

        values: set[int] = set()
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid cron field: '{field}'")

        for part in parts:
            if part.startswith("*/"):
                step = int(part[2:])
                if step < 1:
                    raise ValueError(f"Invalid cron step: '{part}'")
                values.update(range(min_value, max_value + 1, step))
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if allow_seven:
                    if start == 7:
                        start = 0
                    if end == 7:
                        end = 0
                if start > end:
                    raise ValueError(f"Invalid cron range: '{part}'")
                if start < min_value or end > max_value:
                    raise ValueError(f"Cron range out of bounds: '{part}'")
                values.update(range(start, end + 1))
                continue
            number = int(part)
            if allow_seven and number == 7:
                number = 0
            if number < min_value or number > max_value:
                raise ValueError(f"Cron value out of bounds: '{part}'")
            values.add(number)

        if not values:
            raise ValueError(f"Invalid cron field: '{field}'")
        return values, False
