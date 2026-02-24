from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.models import AgentLogEntry
from lsm.ui.shell.commands import agents as agent_commands


class _DummyAgent:
    def __init__(self) -> None:
        self.state = AgentState()

    def run(self, context) -> AgentState:
        self.state.set_status(AgentStatus.RUNNING)
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content="started",
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


def _app(enabled: bool = True):
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=enabled,
                max_tokens_budget=1000,
                sandbox=SimpleNamespace(
                    allowed_read_paths=[],
                    allowed_write_paths=[],
                    allow_url_access=False,
                    require_user_permission={},
                    tool_llm_assignments={},
                ),
            ),
            llm=SimpleNamespace(),
            batch_size=32,
        ),
        query_provider=None,
        query_embedder=None,
    )


def test_handle_agent_command_help_and_status_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    app = _app()

    out = agent_commands.handle_agent_command("/agent", app)
    assert "Agent commands:" in out
    assert "Available agents:" in out
    assert "Academic:" in out

    out2 = agent_commands.handle_agent_command("/agent status", app)
    assert "No active agent." in out2


def test_handle_agent_start_and_controls(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(agent_commands, "create_default_tool_registry", lambda *args, **kwargs: SimpleNamespace(list_definitions=lambda: []))
    monkeypatch.setattr(agent_commands, "ToolSandbox", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(agent_commands, "create_agent", lambda **kwargs: _DummyAgent())

    app = _app(enabled=True)
    start_out = agent_commands.handle_agent_command(
        "/agent start research quantum theory",
        app,
    )
    assert "Started agent 'research'" in start_out

    # Let background thread run.
    time.sleep(0.05)
    status_out = agent_commands.handle_agent_command("/agent status", app)
    assert "Agent: research" in status_out

    pause_out = agent_commands.handle_agent_command("/agent pause", app)
    assert "Paused agent" in pause_out

    resume_out = agent_commands.handle_agent_command("/agent resume", app)
    assert "Resumed agent" in resume_out

    log_out = agent_commands.handle_agent_command("/agent log", app)
    assert "Agent: started" in log_out

    stop_out = agent_commands.handle_agent_command("/agent stop", app)
    assert "Stop requested" in stop_out


def test_handle_agent_start_disabled(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    app = _app(enabled=False)
    out = agent_commands.handle_agent_command("/agent start research ai", app)
    assert "Agents are disabled" in out
