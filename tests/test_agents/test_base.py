from __future__ import annotations

import json
import re as _re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lsm.agents.base import AgentState, AgentStatus, BaseAgent
from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext, AgentLogEntry
from lsm.agents.phase import PhaseResult
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class DummyAgent(BaseAgent):
    name = "dummy"
    description = "Dummy test agent"

    def run(self, initial_context: AgentContext) -> AgentState:
        self.state.set_status(AgentStatus.RUNNING)
        self.state.current_task = "process"
        self.state.add_log(
            AgentLogEntry(
                timestamp=datetime.utcnow(),
                actor="agent",
                content=f"messages={len(initial_context.messages)}",
            )
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state


def test_agent_status_values() -> None:
    assert AgentStatus.IDLE.value == "idle"
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.PAUSED.value == "paused"
    assert AgentStatus.WAITING_USER.value == "waiting_user"
    assert AgentStatus.COMPLETED.value == "completed"
    assert AgentStatus.FAILED.value == "failed"


def test_agent_state_touch_and_log() -> None:
    state = AgentState()
    old_updated_at = state.updated_at - timedelta(seconds=1)
    state.updated_at = old_updated_at
    state.set_status(AgentStatus.RUNNING)
    assert state.status == AgentStatus.RUNNING
    assert state.updated_at > old_updated_at

    entry = AgentLogEntry(timestamp=datetime.utcnow(), actor="agent", content="step")
    state.add_log(entry)
    assert len(state.log_entries) == 1
    assert state.log_entries[0].content == "step"


def test_base_agent_pause_resume_stop() -> None:
    agent = DummyAgent()
    assert agent.state.status == AgentStatus.IDLE

    agent.pause()
    assert agent.state.status == AgentStatus.PAUSED

    agent.resume()
    assert agent.state.status == AgentStatus.RUNNING

    agent.stop()
    assert agent.state.status == AgentStatus.COMPLETED


def test_base_agent_run_updates_state_and_logs() -> None:
    agent = DummyAgent()
    context = AgentContext(messages=[{"role": "user", "content": "hello"}])
    state = agent.run(context)
    assert state.status == AgentStatus.COMPLETED
    assert state.current_task == "process"
    assert len(state.log_entries) == 1
    assert state.log_entries[0].content == "messages=1"


# ---------------------------------------------------------------------------
# Phase 2: workspace accessor and _run_phase() tests
# ---------------------------------------------------------------------------


class _SimpleTool(BaseTool):
    name = "echo"
    description = "Echo input text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("text", ""))


def _base_raw_for_base_tests(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "path": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 5000,
            "max_iterations": 10,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def _make_wired_agent(agent: BaseAgent, tmp_path: Path) -> BaseAgent:
    """Attach runtime attributes required by _run_phase() and workspace accessors."""
    config = build_config_from_raw(_base_raw_for_base_tests(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(_SimpleTool())
    agent.agent_config = config.agents
    agent.llm_registry = config.llm
    agent.tool_registry = registry
    agent.sandbox = ToolSandbox(config.agents.sandbox)
    return agent


# --- workspace accessor tests ---


def test_artifacts_dir_path(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    result = agent._artifacts_dir()
    assert result == Path(str(tmp_path / "Agents" / "dummy" / "artifacts"))
    assert result.is_dir()


def test_logs_dir_path(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    result = agent._logs_dir()
    assert result == Path(str(tmp_path / "Agents" / "dummy" / "logs"))
    assert result.is_dir()


def test_memory_dir_path(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    result = agent._memory_dir()
    assert result == Path(str(tmp_path / "Agents" / "dummy" / "memory"))
    assert result.is_dir()


def test_workspace_root_creates_directories(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    root = agent._workspace_root()
    assert root.is_dir()
    assert (root / "artifacts").is_dir()
    assert (root / "logs").is_dir()
    assert (root / "memory").is_dir()


def test_artifact_filename_basic_format(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    name = agent._artifact_filename("some topic name")
    assert _re.match(r"some_topic_name_\d{8}_\d{6}\.md$", name), f"Unexpected: {name}"


def test_artifact_filename_sanitizes_special_chars(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    name = agent._artifact_filename("topic:one/two")
    assert ":" not in name
    assert "/" not in name
    assert name.endswith(".md")


def test_artifact_filename_strips_leading_trailing_underscores(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    name = agent._artifact_filename(":leading_and_trailing:")
    # ":leading_and_trailing:" → "_leading_and_trailing_" → strip → "leading_and_trailing"
    assert not name.startswith("_")
    # The timestamp portion starts after the name; check the sanitized prefix
    assert name.startswith("leading_and_trailing_")


def test_artifact_filename_custom_suffix(tmp_path: Path) -> None:
    agent = _make_wired_agent(DummyAgent(), tmp_path)
    name = agent._artifact_filename("report", suffix=".json")
    assert name.endswith(".json")
    assert _re.match(r"report_\d{8}_\d{6}\.json$", name), f"Unexpected: {name}"


# --- _run_phase() lifecycle tests ---


class _PhasedAgent(BaseAgent):
    """Agent that drives one _run_phase() call per run()."""

    name = "phased"
    description = "Phased test agent"

    def run(self, initial_context: AgentContext) -> AgentState:
        self._reset_harness()
        self._run_phase(
            system_prompt="Test system prompt",
            user_message="Test message",
        )
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state


class _MultiPhaseAgent(BaseAgent):
    """Agent that calls _run_phase() twice in a single run()."""

    name = "multiphase"
    description = "Multi-phase test agent"

    def run(self, initial_context: AgentContext) -> AgentState:
        self._reset_harness()
        self._run_phase(system_prompt="System", user_message="phase_one")
        self._run_phase(user_message="phase_two")
        self.state.set_status(AgentStatus.COMPLETED)
        return self.state


def _done_resp() -> str:
    return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})


def test_run_phase_creates_harness_on_first_call(monkeypatch, tmp_path: Path) -> None:
    init_count = [0]
    original_init = AgentHarness.__init__

    def spy_init(self_, *args, **kwargs):
        init_count[0] += 1
        original_init(self_, *args, **kwargs)

    monkeypatch.setattr(AgentHarness, "__init__", spy_init)
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: _RecordingProvider([_done_resp()]))

    agent = _make_wired_agent(_MultiPhaseAgent(), tmp_path)
    agent.run(AgentContext(messages=[]))

    assert init_count[0] == 1


def test_run_phase_reuses_harness_on_subsequent_calls(monkeypatch, tmp_path: Path) -> None:
    init_count = [0]
    original_init = AgentHarness.__init__

    def spy_init(self_, *args, **kwargs):
        init_count[0] += 1
        original_init(self_, *args, **kwargs)

    monkeypatch.setattr(AgentHarness, "__init__", spy_init)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: _RecordingProvider([_done_resp(), _done_resp()]),
    )

    agent = _make_wired_agent(_MultiPhaseAgent(), tmp_path)
    agent.run(AgentContext(messages=[]))

    # Two _run_phase() calls in run(), but only one AgentHarness created
    assert init_count[0] == 1


def test_second_run_creates_fresh_harness(monkeypatch, tmp_path: Path) -> None:
    init_count = [0]
    original_init = AgentHarness.__init__

    def spy_init(self_, *args, **kwargs):
        init_count[0] += 1
        original_init(self_, *args, **kwargs)

    monkeypatch.setattr(AgentHarness, "__init__", spy_init)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: _RecordingProvider([_done_resp(), _done_resp()]),
    )

    agent = _make_wired_agent(_PhasedAgent(), tmp_path)
    agent.run(AgentContext(messages=[]))
    agent.run(AgentContext(messages=[]))

    assert init_count[0] == 2


def test_run_phase_returns_phase_result_no_token_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: _RecordingProvider([_done_resp()]))

    agent = _make_wired_agent(_PhasedAgent(), tmp_path)
    agent._reset_harness()
    result = agent._run_phase(system_prompt="sys", user_message="test")

    assert isinstance(result, PhaseResult)
    with pytest.raises(AttributeError):
        _ = result.tokens_used  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = result.cost_usd  # type: ignore[attr-defined]


def test_run_phase_context_label_forwarded_to_run_bounded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: _RecordingProvider([_done_resp(), _done_resp()]),
    )

    agent = _make_wired_agent(_PhasedAgent(), tmp_path)
    agent._reset_harness()
    agent._run_phase(system_prompt="sys", user_message="msg_custom", context_label="custom_ctx")

    assert "custom_ctx" in agent._harness._context_histories


def test_run_phase_direct_tool_calls_no_llm(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_resp()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)

    agent = _make_wired_agent(_PhasedAgent(), tmp_path)
    agent._reset_harness()
    result = agent._run_phase(
        system_prompt="sys",
        direct_tool_calls=[{"name": "echo", "arguments": {"text": "direct"}}],
    )

    assert result.stop_reason == "done"
    assert result.tool_calls[0]["result"] == "direct"
    assert provider.call_count == 0


def test_run_phase_direct_tool_calls_creates_reuses_harness(monkeypatch, tmp_path: Path) -> None:
    init_count = [0]
    original_init = AgentHarness.__init__

    def spy_init(self_, *args, **kwargs):
        init_count[0] += 1
        original_init(self_, *args, **kwargs)

    monkeypatch.setattr(AgentHarness, "__init__", spy_init)
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: _RecordingProvider([]))

    agent = _make_wired_agent(_PhasedAgent(), tmp_path)
    agent._reset_harness()
    agent._run_phase(direct_tool_calls=[{"name": "echo", "arguments": {"text": "a"}}])
    agent._run_phase(direct_tool_calls=[{"name": "echo", "arguments": {"text": "b"}}])

    assert init_count[0] == 1


def test_base_agent_has_no_check_budget_and_stop() -> None:
    agent = DummyAgent()
    with pytest.raises(AttributeError):
        _ = agent._check_budget_and_stop  # type: ignore[attr-defined]


def test_base_agent_has_no_tokens_used() -> None:
    agent = DummyAgent()
    with pytest.raises(AttributeError):
        getattr(agent, "_tokens_used")


class _RecordingProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.call_count = 0

    def _send_message(self, system, user, temperature, max_tokens, **kwargs):
        _ = system, user, temperature, max_tokens, kwargs
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})

