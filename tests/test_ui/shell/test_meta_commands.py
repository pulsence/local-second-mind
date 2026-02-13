from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from lsm.agents.base import AgentState, AgentStatus
from lsm.agents.models import AgentLogEntry
from lsm.agents.task_graph import AgentTask, TaskGraph
from lsm.ui.shell.commands import agents as agent_commands


class _DummyMetaAgent:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.state = AgentState()
        self.last_result = None
        self.last_execution_order: list[str] = []

    def run(self, context) -> AgentState:
        _ = context
        self.state.set_status(AgentStatus.RUNNING)
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content="meta-started",
            )
        )

        self.run_root.mkdir(parents=True, exist_ok=True)
        final_result_path = self.run_root / "final_result.md"
        final_result_path.write_text("# Final Result\n", encoding="utf-8")
        meta_log_path = self.run_root / "meta_log.md"
        meta_log_path.write_text(
            "# Meta Log\n\n## Task Trace\n\n- research_1\n- writing_1\n",
            encoding="utf-8",
        )

        graph = TaskGraph(
            goal="Compose weekly brief",
            tasks=[
                AgentTask(
                    id="research_1",
                    agent_name="research",
                    status="completed",
                ),
                AgentTask(
                    id="writing_1",
                    agent_name="writing",
                    depends_on=["research_1"],
                    status="completed",
                ),
            ],
        )
        self.last_execution_order = ["research_1", "writing_1"]
        self.last_result = SimpleNamespace(
            goal="Compose weekly brief",
            task_graph=graph,
            execution_order=list(self.last_execution_order),
            task_runs=[
                SimpleNamespace(
                    task_id="research_1",
                    agent_name="research",
                    status="completed",
                    sub_agent_dir=str(self.run_root / "sub_agents" / "research_001"),
                    artifacts=[],
                    error=None,
                ),
                SimpleNamespace(
                    task_id="writing_1",
                    agent_name="writing",
                    status="completed",
                    sub_agent_dir=str(self.run_root / "sub_agents" / "writing_001"),
                    artifacts=[str(final_result_path)],
                    error=None,
                ),
            ],
            final_result_path=final_result_path,
            meta_log_path=meta_log_path,
        )
        self.state.add_artifact(str(final_result_path))
        self.state.add_artifact(str(meta_log_path))
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state

    def stop(self) -> None:
        self.state.set_status(AgentStatus.COMPLETED)

    def pause(self) -> None:
        self.state.set_status(AgentStatus.PAUSED)

    def resume(self) -> None:
        self.state.set_status(AgentStatus.RUNNING)


def _app() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            agents=SimpleNamespace(
                enabled=True,
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


def test_meta_commands_start_status_and_log(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    monkeypatch.setattr(
        agent_commands,
        "create_default_tool_registry",
        lambda *args, **kwargs: SimpleNamespace(list_definitions=lambda: []),
    )
    monkeypatch.setattr(agent_commands, "ToolSandbox", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(
        agent_commands,
        "create_agent",
        lambda **kwargs: _DummyMetaAgent(tmp_path / "meta_run"),
    )

    app = _app()
    start_out = agent_commands.handle_agent_command(
        "/agent meta start Compose weekly brief",
        app,
    )
    assert "Started agent 'meta'" in start_out

    time.sleep(0.05)
    status_out = agent_commands.handle_agent_command("/agent meta status", app)
    assert "Meta Agent Status:" in status_out
    assert "Tasks: 2 total, 2 completed" in status_out
    assert "Execution order: research_1, writing_1" in status_out

    log_out = agent_commands.handle_agent_command("/agent meta log", app)
    assert "# Meta Log" in log_out
    assert "## Task Trace" in log_out


def test_meta_commands_validate_usage_and_idle_state(monkeypatch) -> None:
    monkeypatch.setattr(agent_commands, "_MANAGER", agent_commands.AgentRuntimeManager())
    app = _app()

    help_out = agent_commands.handle_agent_command("/agent meta", app)
    assert "Meta-agent commands:" in help_out

    usage_out = agent_commands.handle_agent_command("/agent meta start", app)
    assert "Usage: /agent meta start <goal>" in usage_out

    status_out = agent_commands.handle_agent_command("/agent meta status", app)
    assert "No active meta-agent." in status_out

    log_out = agent_commands.handle_agent_command("/agent meta log", app)
    assert "No active meta-agent." in log_out
