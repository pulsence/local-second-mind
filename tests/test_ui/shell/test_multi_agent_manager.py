from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.models import AgentLogEntry
from lsm.agents.interaction import InteractionRequest
from lsm.ui.shell.commands import agents as agent_commands


class _DummyRegistry:
    def list_definitions(self) -> list[dict]:
        return []


class _DummySandbox:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self.channel = None
        self.waiting_callback = None

    def set_interaction_channel(self, channel, waiting_state_callback=None) -> None:
        self.channel = channel
        self.waiting_callback = waiting_state_callback


class _DummyHarness:
    init_calls: list[dict] = []

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
        interaction_channel=None,
    ) -> None:
        _ = (
            agent_config,
            tool_registry,
            llm_registry,
            sandbox,
            agent_name,
            tool_allowlist,
            vectordb_config,
            memory_store,
            memory_context_builder,
        )
        self.interaction_channel = interaction_channel
        self.stopped = False
        self.paused = False
        _DummyHarness.init_calls.append(
            {
                "agent_name": agent_name,
                "interaction_channel": interaction_channel,
            }
        )

    def stop(self) -> None:
        self.stopped = True

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False


class _LoopAgent:
    def __init__(self) -> None:
        self.state = AgentState()
        self._stop_event = threading.Event()
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        while not self._stop_event.wait(timeout=0.01):
            continue
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop_event.set()
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.pause_calls += 1
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.resume_calls += 1
        self.state.set_status(AgentStatus.RUNNING)


class _InteractionAgent:
    def __init__(self, channel) -> None:
        self.state = AgentState()
        self._channel = channel
        self.stop_calls = 0

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        response = self._channel.post_request(
            InteractionRequest(
                request_id="clarify-1",
                request_type="clarification",
                reason="Need user answer",
                prompt="Proceed?",
            )
        )
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content=f"decision={response.decision}",
            )
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.stop_calls += 1
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


class _RepeatInteractionAgent:
    def __init__(self, channel) -> None:
        self.state = AgentState()
        self._channel = channel

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        try:
            self._channel.post_request(
                InteractionRequest(
                    request_id="perm-1",
                    request_type="permission",
                    tool_name="write_file",
                    reason="Need approval",
                    prompt="Allow write_file?",
                )
            )
        except Exception:
            pass
        # This second request should be blocked once stop() closes the channel.
        try:
            self._channel.post_request(
                InteractionRequest(
                    request_id="perm-2",
                    request_type="permission",
                    tool_name="write_file",
                    reason="Need approval again",
                    prompt="Allow write_file again?",
                )
            )
        except Exception:
            pass
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


class _SinglePermissionAgent:
    def __init__(self, channel, request_id: str = "perm-1") -> None:
        self.state = AgentState()
        self._channel = channel
        self._request_id = request_id

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        response = self._channel.post_request(
            InteractionRequest(
                request_id=self._request_id,
                request_type="permission",
                tool_name="write_file",
                reason="Need approval",
                prompt="Allow write_file?",
            )
        )
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content=f"decision={response.decision}",
            )
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


class _StopAwareAgent:
    def __init__(
        self,
        *,
        log_path: Path,
        action_delay_s: float = 0.3,
    ) -> None:
        self.state = AgentState()
        self._log_path = Path(log_path)
        self._action_delay_s = max(0.0, float(action_delay_s))
        self._stop_requested = threading.Event()
        self.action_started = threading.Event()

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        # Simulate a currently running action that should be allowed to finish.
        self.action_started.set()
        time.sleep(self._action_delay_s)
        if not self._stop_requested.is_set():
            time.sleep(0.05)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text("saved", encoding="utf-8")
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self._stop_requested.set()

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


class _BurstLogAgent:
    def __init__(self, *, count: int = 8) -> None:
        self.state = AgentState()
        self._count = max(1, int(count))

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        for idx in range(self._count):
            self.state.add_log(
                AgentLogEntry(
                    timestamp=datetime.utcnow(),
                    actor="tool",
                    content=f"msg-{idx}",
                    action="echo",
                    action_arguments={"index": idx},
                )
            )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


def _app(
    max_concurrent: int = 2,
    *,
    log_stream_queue_limit: int = 500,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=True,
                max_tokens_budget=1000,
                max_concurrent=max_concurrent,
                log_stream_queue_limit=log_stream_queue_limit,
                memory=SimpleNamespace(enabled=False),
                interaction=SimpleNamespace(timeout_seconds=2, timeout_action="deny"),
                sandbox=SimpleNamespace(
                    allowed_read_paths=[],
                    allowed_write_paths=[],
                    allow_url_access=False,
                    require_user_permission={},
                    tool_llm_assignments={},
                ),
            ),
            llm=SimpleNamespace(),
            vectordb=SimpleNamespace(),
            batch_size=32,
        ),
        query_provider=None,
        query_embedder=None,
    )


def _extract_agent_id(output: str) -> str:
    match = re.search(r"id=([0-9a-f]+)", output)
    assert match is not None, output
    return match.group(1)


def _wait_until(predicate, timeout_s: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_multi_agent_start_enforces_max_concurrent(monkeypatch) -> None:
    agents: list[_LoopAgent] = []
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)

    def _create_agent(**kwargs):
        _ = kwargs
        agent = _LoopAgent()
        agents.append(agent)
        return agent

    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    manager = agent_commands.AgentRuntimeManager()
    app = _app(max_concurrent=2)

    start_1 = manager.start(app, "research", "topic-1")
    start_2 = manager.start(app, "writing", "topic-2")
    assert "Started agent 'research'" in start_1
    assert "Started agent 'writing'" in start_2
    assert len(manager.list_running()) == 2

    rejected = manager.start(app, "synthesis", "topic-3")
    assert "max_concurrent limit (2) reached" in rejected

    manager.shutdown(join_timeout_s=1.0)
    assert len(manager.list_running()) == 0
    assert all(agent.stop_calls >= 1 for agent in agents)


def test_multi_agent_controls_target_by_id_and_keep_single_agent_compat(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(agent_commands, "create_agent", lambda **kwargs: _LoopAgent())

    manager = agent_commands.AgentRuntimeManager()
    app = _app(max_concurrent=3)

    id_1 = _extract_agent_id(manager.start(app, "research", "a"))
    id_2 = _extract_agent_id(manager.start(app, "writing", "b"))

    # Multi-agent default targeting uses selected agent id (latest start or /agent select).
    assert f"({id_2})" in manager.pause()
    assert f"({id_1})" in manager.pause(agent_id=id_1)
    assert f"({id_1})" in manager.resume(agent_id=id_1)

    with manager._lock:
        agent_1 = manager._agents[id_1].agent
        agent_2 = manager._agents[id_2].agent
    assert agent_1.pause_calls == 1
    assert agent_2.pause_calls == 1

    assert f"({id_1})" in manager.stop(agent_id=id_1)
    assert _wait_until(lambda: len(manager.list_running()) == 1)

    # Backward compatibility: with one active agent, no-id control actions target it.
    pause_out = manager.pause()
    assert "Paused agent" in pause_out
    resume_out = manager.resume()
    assert "Resumed agent" in resume_out

    manager.shutdown(join_timeout_s=1.0)


def test_multi_agent_pending_interactions_and_response_flow(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)

    def _create_agent(**kwargs):
        sandbox = kwargs["sandbox"]
        return _InteractionAgent(sandbox.channel)

    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    manager = agent_commands.AgentRuntimeManager()
    app = _app(max_concurrent=1)
    agent_id = _extract_agent_id(manager.start(app, "research", "needs-answer"))

    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)
    pending = manager.get_pending_interaction(agent_id=agent_id)
    assert pending is not None
    assert pending["request_type"] == "clarification"
    assert pending["prompt"] == "Proceed?"

    reply_out = manager.respond_to_interaction(
        agent_id,
        {
            "decision": "reply",
            "user_message": "yes",
        },
    )
    assert "Posted interaction response" in reply_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)

    status_out = manager.status(agent_id=agent_id)
    assert "Status: completed" in status_out
    log_out = manager.log(agent_id=agent_id)
    assert "decision=reply" in log_out


def test_multi_agent_shutdown_cancels_pending_and_joins(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(
        agent_commands,
        "create_agent",
        lambda **kwargs: _InteractionAgent(kwargs["sandbox"].channel),
    )

    manager = agent_commands.AgentRuntimeManager()
    app = _app(max_concurrent=1)
    agent_id = _extract_agent_id(manager.start(app, "research", "blocking"))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)

    manager.shutdown(join_timeout_s=1.0)
    assert manager.list_running() == []
    assert manager.get_pending_interactions() == []
    assert "Status: completed" in manager.status(agent_id=agent_id)


def test_multi_agent_stop_closes_channel_and_prevents_reprompt(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(
        agent_commands,
        "create_agent",
        lambda **kwargs: _RepeatInteractionAgent(kwargs["sandbox"].channel),
    )

    manager = agent_commands.AgentRuntimeManager(join_timeout_s=1.0)
    app = _app(max_concurrent=1)
    agent_id = _extract_agent_id(manager.start(app, "research", "blocking"))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)

    stop_out = manager.stop(agent_id=agent_id)
    assert "Stop requested for agent" in stop_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    assert manager.get_pending_interactions() == []


def test_multi_agent_approve_session_persists_across_runs(monkeypatch) -> None:
    counter = {"idx": 0}

    def _create_agent(**kwargs):
        counter["idx"] += 1
        return _SinglePermissionAgent(
            kwargs["sandbox"].channel,
            request_id=f"perm-{counter['idx']}",
        )

    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    manager = agent_commands.AgentRuntimeManager(join_timeout_s=1.0)
    app = _app(max_concurrent=1)

    first_id = _extract_agent_id(manager.start(app, "research", "first"))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)
    approve_out = manager.respond_to_interaction(
        first_id,
        {
            "decision": "approve_session",
        },
    )
    assert "Posted interaction response" in approve_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)

    second_id = _extract_agent_id(manager.start(app, "research", "second"))
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    assert manager.get_pending_interactions() == []
    second_log = manager.log(agent_id=second_id)
    assert "decision=approve_session" in second_log


def test_multi_agent_stop_waits_for_current_action_and_log_persist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)

    holder: dict[str, _StopAwareAgent] = {}
    log_path = tmp_path / "stop-log.txt"

    def _create_agent(**kwargs):
        _ = kwargs
        agent = _StopAwareAgent(log_path=log_path, action_delay_s=0.3)
        holder["agent"] = agent
        return agent

    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    manager = agent_commands.AgentRuntimeManager(join_timeout_s=0.1)
    app = _app(max_concurrent=1)
    agent_id = _extract_agent_id(manager.start(app, "research", "stop-aware"))
    assert _wait_until(lambda: "agent" in holder and holder["agent"].action_started.is_set())

    stop_out = manager.stop(agent_id=agent_id)
    assert "still running" not in stop_out.lower()
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    assert log_path.exists()


def test_multi_agent_completed_history_pruning(monkeypatch) -> None:
    class _FastAgent:
        def __init__(self) -> None:
            self.state = AgentState()

        def run(self, context) -> AgentState:
            _ = context
            self.state.set_status(AgentStatus.COMPLETED)
            return self.state

        def stop(self) -> None:
            self.state.set_status(AgentStatus.COMPLETED)

        def pause(self) -> None:
            self.state.set_status(AgentStatus.PAUSED)

        def resume(self) -> None:
            self.state.set_status(AgentStatus.RUNNING)

    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(agent_commands, "create_agent", lambda **kwargs: _FastAgent())

    manager = agent_commands.AgentRuntimeManager(completed_retention=2)
    app = _app(max_concurrent=2)

    ids: list[str] = []
    for idx in range(3):
        ids.append(_extract_agent_id(manager.start(app, "research", f"topic-{idx}")))
        assert _wait_until(lambda: len(manager.list_running()) == 0)

    with manager._lock:
        retained = set(manager._completed_runs.keys())
    assert len(retained) == 2
    assert ids[0] not in retained
    assert ids[1] in retained
    assert ids[2] in retained


def test_multi_agent_log_stream_drains_entries(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(
        agent_commands,
        "create_agent",
        lambda **kwargs: _BurstLogAgent(count=4),
    )

    manager = agent_commands.AgentRuntimeManager(join_timeout_s=1.0)
    app = _app(max_concurrent=1, log_stream_queue_limit=32)

    agent_id = _extract_agent_id(manager.start(app, "research", "stream"))
    assert _wait_until(lambda: len(manager.list_running()) == 0)

    payload = manager.drain_log_stream(agent_id, max_entries=10)
    assert payload["agent_id"] == agent_id
    entries = payload["entries"]
    assert len(entries) == 4
    assert entries[0]["action"] == "echo"
    assert entries[-1]["content"] == "msg-3"
    assert payload["dropped_count"] == 0

    empty_payload = manager.drain_log_stream(agent_id, max_entries=10)
    assert empty_payload["entries"] == []
    assert empty_payload["dropped_count"] == 0


def test_multi_agent_log_stream_backpressure_reports_drops(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(
        agent_commands,
        "create_agent",
        lambda **kwargs: _BurstLogAgent(count=7),
    )

    manager = agent_commands.AgentRuntimeManager(join_timeout_s=1.0)
    app = _app(max_concurrent=1, log_stream_queue_limit=2)

    agent_id = _extract_agent_id(manager.start(app, "research", "backpressure"))
    assert _wait_until(lambda: len(manager.list_running()) == 0)

    payload = manager.drain_log_stream(agent_id, max_entries=10)
    entries = payload["entries"]
    assert len(entries) == 2
    assert payload["dropped_count"] >= 5
    assert entries[0]["content"] == "msg-5"
    assert entries[1]["content"] == "msg-6"
