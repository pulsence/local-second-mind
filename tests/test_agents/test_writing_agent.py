from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from lsm.agents.base import BaseAgent
from lsm.agents.factory import create_agent
from lsm.agents.models import AgentContext
from lsm.agents.phase import PhaseResult
from lsm.agents.productivity import WritingAgent
from lsm.agents.tools.base import ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


def _base_raw(tmp_path: Path) -> dict:
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
            "max_tokens_budget": 15000,
            "max_iterations": 4,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {"writing": {"max_iterations": 2}},
        },
    }


def _make_result(
    final_text: str,
    stop_reason: str = "stop",
    tool_calls: list | None = None,
) -> PhaseResult:
    return PhaseResult(
        final_text=final_text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
    )


def _build_agent(tmp_path: Path) -> WritingAgent:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    return WritingAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_writing_agent_runs_and_saves_deliverable(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline\n\n## Intro\n\n## Body\n"),  # OUTLINE
        _make_result("# Draft Deliverable\n\nInitial grounded draft.\n"),  # DRAFT
        _make_result("# Final Deliverable\n\nRevised grounded draft.\n"),  # REVIEW
    ]):
        state = agent.run(
            AgentContext(messages=[{"role": "user", "content": "Write on sacramental realism"}])
        )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.log_path.exists()
    content = agent.last_result.output_path.read_text(encoding="utf-8")
    assert "# Final Deliverable" in content

    saved_log = agent.last_result.log_path.read_text(encoding="utf-8")
    assert saved_log.strip()


def test_writing_agent_phases_execute_in_order(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    phase_messages: list[str] = []

    def capture(**kwargs):
        phase_messages.append(kwargs.get("user_message", ""))
        return _make_result("ok")

    with patch.object(BaseAgent, "_run_phase", side_effect=capture):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert len(phase_messages) == 3
    assert "OUTLINE" in phase_messages[0]
    assert "DRAFT" in phase_messages[1]
    assert "REVIEW" in phase_messages[2]


def test_writing_agent_outline_phase_uses_query_knowledge_base(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline"),
        _make_result("# Draft"),
        _make_result("# Review"),
    ]) as mock_phase:
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    calls = mock_phase.call_args_list
    assert calls[0].kwargs.get("tool_names") == ["query_knowledge_base"]


def test_writing_agent_draft_and_review_use_no_tools(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline"),
        _make_result("# Draft"),
        _make_result("# Review"),
    ]) as mock_phase:
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    calls = mock_phase.call_args_list
    assert calls[1].kwargs.get("tool_names") == []
    assert calls[2].kwargs.get("tool_names") == []


def test_writing_agent_output_in_artifacts_dir(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline"),
        _make_result("# Draft"),
        _make_result("# Final\n\nContent.\n"),
    ]):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert agent.last_result is not None
    assert agent.last_result.output_path.parent.name == "artifacts"
    assert agent.last_result.output_path.suffix == ".md"


def test_writing_agent_skips_draft_review_on_budget_exhausted(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline", stop_reason="budget_exhausted"),  # OUTLINE exhausts budget
        # DRAFT and REVIEW must NOT be called
    ]) as mock_phase:
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert mock_phase.call_count == 1


def test_writing_agent_has_no_tokens_used_attribute(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline"),
        _make_result("# Draft"),
        _make_result("# Review"),
    ]):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    with pytest.raises(AttributeError):
        getattr(agent, "_tokens_used")


# ---------------------------------------------------------------------------
# Factory test
# ---------------------------------------------------------------------------

def test_agent_factory_creates_writing_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent(
        "writing",
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )
    assert isinstance(agent, WritingAgent)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("# Outline"),
        _make_result("# Draft"),
        _make_result("# Final deliverable\n"),
    ]):
        state = agent.run(
            AgentContext(messages=[{"role": "user", "content": "Factory writing topic"}])
        )

    assert state.status.value == "completed"


# ---------------------------------------------------------------------------
# Source-inspection tests
# ---------------------------------------------------------------------------

def test_writing_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.productivity.writing as writing_module

    source = inspect.getsource(writing_module)
    assert "AgentHarness(" not in source


def test_writing_agent_has_no_direct_provider_call() -> None:
    import lsm.agents.productivity.writing as writing_module

    source = inspect.getsource(writing_module)
    assert "provider.synthesize(" not in source
    assert "provider.complete(" not in source


def test_writing_agent_has_no_direct_sandbox_execute_call() -> None:
    import lsm.agents.productivity.writing as writing_module

    source = inspect.getsource(writing_module)
    assert "sandbox.execute(" not in source
