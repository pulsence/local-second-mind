from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from types import SimpleNamespace

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.interaction import InteractionRequest
from lsm.agents.models import AgentLogEntry
from lsm.ui.shell.commands import agents as agent_commands


class _DummyRegistry:
    def list_definitions(self) -> list[dict]:
        return []


class _DummySandbox:
    def __init__(self, cfg) -> None:
        self.config = cfg
        self.channel = None

    def set_interaction_channel(self, channel, waiting_state_callback=None) -> None:
        _ = waiting_state_callback
        self.channel = channel


class _DummyHarness:
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
            interaction_channel,
        )

    def stop(self) -> None:
        return

    def pause(self) -> None:
        return

    def resume(self) -> None:
        return


class _LoopAgent:
    def __init__(self) -> None:
        self.state = AgentState()
        self._stop_event = threading.Event()
        self.pause_calls = 0
        self.resume_calls = 0

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        while not self._stop_event.wait(timeout=0.01):
            continue
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self._stop_event.set()
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.pause_calls += 1
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.resume_calls += 1
        self.state.set_status(AgentStatus.RUNNING)


class _InteractionAgent:
    def __init__(
        self,
        channel,
        *,
        request_type: str,
        request_id: str,
        tool_name: str | None = None,
    ) -> None:
        self.state = AgentState()
        self._channel = channel
        self._request_type = request_type
        self._request_id = request_id
        self._tool_name = tool_name

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        response = self._channel.post_request(
            InteractionRequest(
                request_id=self._request_id,
                request_type=self._request_type,
                tool_name=self._tool_name,
                reason="Need user response",
                prompt="Continue?",
            )
        )
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content=(
                    f"decision={response.decision} "
                    f"message={response.user_message}"
                ),
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


def _app(max_concurrent: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=True,
                max_tokens_budget=1000,
                max_concurrent=max_concurrent,
                memory=SimpleNamespace(enabled=False),
                interaction=SimpleNamespace(timeout_seconds=3, timeout_action="deny"),
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


def _wait_until(predicate, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_agent_cli_list_and_select_target_selected_agent(monkeypatch) -> None:
    manager = agent_commands.AgentRuntimeManager(join_timeout_s=0.5)
    monkeypatch.setattr(agent_commands, "_MANAGER", manager)
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(agent_commands, "create_agent", lambda **kwargs: _LoopAgent())

    app = _app(max_concurrent=2)
    id_1 = _extract_agent_id(agent_commands.handle_agent_command("/agent start research alpha", app))
    id_2 = _extract_agent_id(agent_commands.handle_agent_command("/agent start writing beta", app))
    assert id_1 != id_2

    list_out = agent_commands.handle_agent_command("/agent list", app)
    assert "Running agents (2):" in list_out
    assert id_1[:8] in list_out
    assert id_2[:8] in list_out

    status_out = agent_commands.handle_agent_command("/agent status", app)
    assert "Agents: 2 active" in status_out

    select_out = agent_commands.handle_agent_command(f"/agent select {id_1[:8]}", app)
    assert f"({id_1})" in select_out

    pause_out = agent_commands.handle_agent_command("/agent pause", app)
    assert f"({id_1})" in pause_out
    with manager._lock:
        first_agent = manager._agents[id_1].agent
        second_agent = manager._agents[id_2].agent
    assert first_agent.pause_calls == 1
    assert second_agent.pause_calls == 0

    agent_commands.handle_agent_command(f"/agent stop {id_1}", app)
    agent_commands.handle_agent_command(f"/agent stop {id_2}", app)
    manager.shutdown(join_timeout_s=0.5)


def test_agent_cli_interact_approve_and_deny(monkeypatch) -> None:
    manager = agent_commands.AgentRuntimeManager(join_timeout_s=0.5)
    monkeypatch.setattr(agent_commands, "_MANAGER", manager)
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)

    counter = {"idx": 0}

    def _create_agent(**kwargs):
        counter["idx"] += 1
        return _InteractionAgent(
            kwargs["sandbox"].channel,
            request_type="permission",
            request_id=f"perm-{counter['idx']}",
            tool_name="write_file",
        )

    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    app = _app(max_concurrent=1)
    approve_id = _extract_agent_id(agent_commands.handle_agent_command("/agent start research approve-me", app))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)

    interact_out = agent_commands.handle_agent_command("/agent interact", app)
    assert "Pending interactions (1):" in interact_out
    assert approve_id[:8] in interact_out
    assert "type=permission" in interact_out

    approve_out = agent_commands.handle_agent_command(f"/agent approve {approve_id[:8]}", app)
    assert "Posted interaction response" in approve_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    approve_log = agent_commands.handle_agent_command(f"/agent log {approve_id}", app)
    assert "decision=approve" in approve_log

    deny_id = _extract_agent_id(agent_commands.handle_agent_command("/agent start research deny-me", app))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)
    deny_out = agent_commands.handle_agent_command(
        f'/agent deny {deny_id[:8]} "not now"',
        app,
    )
    assert "Posted interaction response" in deny_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    deny_log = agent_commands.handle_agent_command(f"/agent log {deny_id}", app)
    assert "decision=deny" in deny_log
    assert "message=not now" in deny_log


def test_agent_cli_reply_and_approve_session(monkeypatch) -> None:
    manager = agent_commands.AgentRuntimeManager(join_timeout_s=0.5)
    monkeypatch.setattr(agent_commands, "_MANAGER", manager)
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)

    mode = {"type": "clarification"}
    counter = {"idx": 0}

    def _create_agent(**kwargs):
        counter["idx"] += 1
        request_type = mode["type"]
        tool_name = "write_file" if request_type == "permission" else None
        return _InteractionAgent(
            kwargs["sandbox"].channel,
            request_type=request_type,
            request_id=f"req-{counter['idx']}",
            tool_name=tool_name,
        )

    monkeypatch.setattr(agent_commands, "create_agent", _create_agent)

    app = _app(max_concurrent=1)

    reply_id = _extract_agent_id(agent_commands.handle_agent_command("/agent start research clarify", app))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)
    reply_out = agent_commands.handle_agent_command(
        f'/agent reply {reply_id[:8]} "Use local notes"',
        app,
    )
    assert "Posted interaction response" in reply_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    reply_log = agent_commands.handle_agent_command(f"/agent log {reply_id}", app)
    assert "decision=reply" in reply_log
    assert "message=Use local notes" in reply_log

    mode["type"] = "permission"
    session_id = _extract_agent_id(agent_commands.handle_agent_command("/agent start research permission-1", app))
    assert _wait_until(lambda: len(manager.get_pending_interactions()) == 1)
    approve_session_out = agent_commands.handle_agent_command(
        f"/agent approve-session {session_id[:8]}",
        app,
    )
    assert "Posted interaction response" in approve_session_out
    assert _wait_until(lambda: len(manager.list_running()) == 0)

    second_id = _extract_agent_id(agent_commands.handle_agent_command("/agent start research permission-2", app))
    assert _wait_until(lambda: len(manager.list_running()) == 0)
    assert manager.get_pending_interactions() == []
    second_log = agent_commands.handle_agent_command(f"/agent log {second_id}", app)
    assert "decision=approve_session" in second_log


def test_agent_cli_queue_and_resume_with_message(monkeypatch) -> None:
    manager = agent_commands.AgentRuntimeManager(join_timeout_s=0.5)
    monkeypatch.setattr(agent_commands, "_MANAGER", manager)
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: _DummyRegistry())
    monkeypatch.setattr(agent_commands, "ToolSandbox", _DummySandbox)
    monkeypatch.setattr(agent_commands, "AgentHarness", _DummyHarness)
    monkeypatch.setattr(agent_commands, "create_agent", lambda **kwargs: _LoopAgent())

    app = _app(max_concurrent=1)
    agent_id = _extract_agent_id(
        agent_commands.handle_agent_command("/agent start research queued-message", app)
    )

    pause_out = agent_commands.handle_agent_command("/agent pause", app)
    assert f"({agent_id})" in pause_out

    queue_out = agent_commands.handle_agent_command(
        '/agent queue "Focus on local notes first"',
        app,
    )
    assert "Queued user message for agent" in queue_out
    with manager._lock:
        queued_context = list(manager._agents[agent_id].context.messages)
    assert queued_context[-1] == {
        "role": "user",
        "content": "Focus on local notes first",
    }

    resume_out = agent_commands.handle_agent_command(
        '/agent resume "Continue and prioritize recent files"',
        app,
    )
    assert "Queued user message for agent" in resume_out
    assert "Resumed agent" in resume_out
    with manager._lock:
        resumed_context = list(manager._agents[agent_id].context.messages)
    assert resumed_context[-1] == {
        "role": "user",
        "content": "Continue and prioritize recent files",
    }

    log_out = agent_commands.handle_agent_command(f"/agent log {agent_id}", app)
    assert "Focus on local notes first" in log_out
    assert "Continue and prioritize recent files" in log_out

    agent_commands.handle_agent_command(f"/agent stop {agent_id}", app)
    manager.shutdown(join_timeout_s=0.5)
