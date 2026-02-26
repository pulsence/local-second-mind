from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.scheduler import AgentScheduler
from lsm.config.loader import build_config_from_raw


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance(self, **kwargs) -> datetime:
        self.current = self.current + timedelta(**kwargs)
        return self.current


class FakeHarness:
    _lock = threading.Lock()
    init_calls: list[dict] = []
    instances: list["FakeHarness"] = []
    run_started = threading.Event()
    release_run = threading.Event()
    block_runs = False
    total_runs = 0
    active_runs = 0
    max_active_runs = 0

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.init_calls = []
            cls.instances = []
            cls.total_runs = 0
            cls.active_runs = 0
            cls.max_active_runs = 0
            cls.block_runs = False
        cls.run_started = threading.Event()
        cls.release_run = threading.Event()

    def __init__(
        self,
        agent_config,
        tool_registry,
        llm_registry,
        sandbox,
        agent_name="agent",
        tool_allowlist=None,
        vectordb_config=None,
        memory_store=None,
        memory_context_builder=None,
    ) -> None:
        _ = tool_registry, llm_registry, vectordb_config, memory_store, memory_context_builder
        self.agent_config = agent_config
        self.sandbox = sandbox
        self.agent_name = agent_name
        self.tool_allowlist = set(tool_allowlist or set())
        self.stop_called = False
        with self._lock:
            self.instances.append(self)
            self.init_calls.append(
                {
                    "agent_name": agent_name,
                    "tool_allowlist": set(self.tool_allowlist),
                    "sandbox_allow_url_access": sandbox.config.allow_url_access,
                    "sandbox_allowed_write_paths": list(sandbox.config.allowed_write_paths),
                    "sandbox_execution_mode": sandbox.config.execution_mode,
                    "sandbox_force_docker": sandbox.config.force_docker,
                }
            )

    def run(self, initial_context) -> AgentState:
        _ = initial_context
        with self._lock:
            self.__class__.total_runs += 1
            self.__class__.active_runs += 1
            self.__class__.max_active_runs = max(
                self.__class__.max_active_runs,
                self.__class__.active_runs,
            )
        self.__class__.run_started.set()
        try:
            while self.__class__.block_runs:
                if self.stop_called:
                    break
                if self.__class__.release_run.wait(timeout=0.01):
                    break
        finally:
            with self._lock:
                self.__class__.active_runs = max(0, self.__class__.active_runs - 1)

        state = AgentState()
        state.set_status(AgentStatus.COMPLETED)
        return state

    def stop(self) -> None:
        self.stop_called = True


@pytest.fixture(autouse=True)
def _reset_fake_harness() -> None:
    FakeHarness.reset()
    yield
    FakeHarness.release_run.set()


def _fake_agent_factory(**kwargs):
    return SimpleNamespace(
        tool_allowlist={
            "read_file",
            "read_folder",
            "file_metadata",
            "hash_file",
            "source_map",
            "query_knowledge_base",
            "extract_snippets",
            "write_file",
            "append_file",
            "create_folder",
            "load_url",
            "query_llm",
            "query_remote",
            "query_remote_chain",
            "similarity_search",
        },
        agent_config=kwargs["agent_config"],
    )


def _base_raw(tmp_path: Path, schedule: dict) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 5000,
            "max_iterations": 5,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path / "out")],
                "allow_url_access": True,
                "require_user_permission": {},
                "require_permission_by_risk": {},
                "execution_mode": "local_only",
                "limits": {
                    "timeout_s_default": 10,
                    "max_stdout_kb": 128,
                    "max_file_write_mb": 2,
                },
                "docker": {
                    "enabled": True,
                    "image": "lsm-agent-sandbox:latest",
                    "network_default": "none",
                    "cpu_limit": 1.0,
                    "mem_limit_mb": 256,
                    "read_only_root": True,
                },
                "tool_llm_assignments": {},
            },
            "memory": {"enabled": False},
            "schedules": [schedule],
        },
    }


def _create_scheduler(tmp_path: Path, schedule: dict, clock: FakeClock) -> AgentScheduler:
    config = build_config_from_raw(_base_raw(tmp_path, schedule), tmp_path / "config.json")
    return AgentScheduler(
        config,
        tick_seconds=1.0,
        now_fn=clock.now,
        harness_cls=FakeHarness,
        agent_factory=_fake_agent_factory,
    )


def test_scheduler_persists_and_reloads_schedule_state(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "daily recap"},
            "interval": "1s",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        assert scheduler.wait_until_idle(timeout_s=2.0)

        snapshots = scheduler.list_schedules()
        assert len(snapshots) == 1
        assert snapshots[0]["last_status"] == "completed"
        assert snapshots[0]["last_run_at"] is not None

        state_path = scheduler.state_path
        assert state_path.exists()
        persisted = json.loads(state_path.read_text(encoding="utf-8"))
        assert persisted["schedules"][0]["last_status"] == "completed"

        reloaded = _create_scheduler(
            tmp_path,
            {
                "agent_name": "research",
                "params": {"topic": "daily recap"},
                "interval": "1s",
                "enabled": True,
                "concurrency_policy": "skip",
                "confirmation_mode": "auto",
            },
            clock,
        )
        try:
            reloaded_snapshots = reloaded.list_schedules()
            assert reloaded_snapshots[0]["last_status"] == "completed"
            assert reloaded_snapshots[0]["last_run_at"] == snapshots[0]["last_run_at"]
        finally:
            reloaded.stop()
    finally:
        scheduler.stop()


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_queue", "expect_stop_called"),
    [
        ("skip", "skipped", 0, False),
        ("queue", "queued", 1, False),
        ("cancel", "cancel_requested", 1, True),
    ],
)
def test_scheduler_overlap_policies_prevent_concurrent_runs(
    tmp_path: Path,
    policy: str,
    expected_status: str,
    expected_queue: int,
    expect_stop_called: bool,
) -> None:
    FakeHarness.block_runs = True
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "overlap"},
            "interval": "1s",
            "enabled": True,
            "concurrency_policy": policy,
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        assert FakeHarness.run_started.wait(timeout=1.0)
        assert FakeHarness.total_runs == 1

        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        snapshots = scheduler.list_schedules()
        assert snapshots[0]["last_status"] == expected_status
        assert snapshots[0]["queued_runs"] == expected_queue
        assert FakeHarness.max_active_runs <= 1

        if expect_stop_called:
            assert FakeHarness.instances[0].stop_called is True

        FakeHarness.release_run.set()
        assert scheduler.wait_until_idle(timeout_s=2.0)
        assert FakeHarness.max_active_runs <= 1
    finally:
        scheduler.stop()


def test_scheduler_defaults_to_read_only_and_no_network(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "safe defaults"},
            "interval": "1s",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        assert scheduler.wait_until_idle(timeout_s=2.0)

        init = FakeHarness.init_calls[0]
        allowlist = init["tool_allowlist"]
        assert "write_file" not in allowlist
        assert "create_folder" not in allowlist
        assert "load_url" not in allowlist
        assert "query_remote" not in allowlist
        assert "query_llm" not in allowlist
        assert init["sandbox_allow_url_access"] is False
        assert init["sandbox_allowed_write_paths"] == []
        assert init["sandbox_execution_mode"] == "local_only"
        assert init["sandbox_force_docker"] is False
    finally:
        scheduler.stop()


def test_scheduler_requires_opt_in_for_writes_and_network(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {
                "topic": "unsafe ops",
                "allow_writes": True,
                "allow_network": True,
            },
            "interval": "1s",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        assert scheduler.wait_until_idle(timeout_s=2.0)

        init = FakeHarness.init_calls[0]
        allowlist = init["tool_allowlist"]
        assert "write_file" in allowlist
        assert "load_url" in allowlist
        assert "query_remote" in allowlist
        assert init["sandbox_allow_url_access"] is True
        assert len(init["sandbox_allowed_write_paths"]) == 1
        assert init["sandbox_execution_mode"] == "prefer_docker"
        assert init["sandbox_force_docker"] is False
    finally:
        scheduler.stop()


def test_scheduler_force_docker_param_enables_force_mode(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {
                "topic": "unsupervised hardening",
                "force_docker": True,
            },
            "interval": "1s",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        clock.advance(seconds=2)
        scheduler.run_pending_once(now=clock.now())
        assert scheduler.wait_until_idle(timeout_s=2.0)

        init = FakeHarness.init_calls[0]
        assert init["sandbox_execution_mode"] == "prefer_docker"
        assert init["sandbox_force_docker"] is True
    finally:
        scheduler.stop()


def test_scheduler_tick_logic_only_runs_due_schedules(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "timing"},
            "interval": "2s",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        initial = scheduler.list_schedules()[0]
        first_due = datetime.fromisoformat(initial["next_run_at"])

        scheduler.run_pending_once(now=first_due - timedelta(seconds=1))
        assert FakeHarness.total_runs == 0

        scheduler.run_pending_once(now=first_due)
        assert scheduler.wait_until_idle(timeout_s=2.0)
        assert FakeHarness.total_runs == 1

        after = scheduler.list_schedules()[0]
        second_due = datetime.fromisoformat(after["next_run_at"])
        assert second_due > first_due
    finally:
        scheduler.stop()


def test_scheduler_runs_due_cron_schedule(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 2, 13, 12, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "cron cadence"},
            "interval": "*/5 * * * * *",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        initial = scheduler.list_schedules()[0]
        first_due = datetime.fromisoformat(initial["next_run_at"])
        assert first_due == datetime(2026, 2, 13, 12, 0, 5, tzinfo=timezone.utc)

        scheduler.run_pending_once(now=first_due - timedelta(seconds=1))
        assert FakeHarness.total_runs == 0

        scheduler.run_pending_once(now=first_due)
        assert scheduler.wait_until_idle(timeout_s=2.0)
        assert FakeHarness.total_runs == 1
    finally:
        scheduler.stop()


def test_scheduler_cron_leap_day_expression_computes_next_run(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc))
    scheduler = _create_scheduler(
        tmp_path,
        {
            "agent_name": "research",
            "params": {"topic": "leap day"},
            "interval": "0 0 29 2 *",
            "enabled": True,
            "concurrency_policy": "skip",
            "confirmation_mode": "auto",
        },
        clock,
    )
    try:
        initial = scheduler.list_schedules()[0]
        leap_due = datetime.fromisoformat(initial["next_run_at"])
        assert leap_due == datetime(2028, 2, 29, 0, 0, 0, tzinfo=timezone.utc)

        scheduler.run_pending_once(now=leap_due)
        assert scheduler.wait_until_idle(timeout_s=2.0)
        assert FakeHarness.total_runs == 1
    finally:
        scheduler.stop()
