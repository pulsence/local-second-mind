from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

from lsm.agents.base import BaseAgent
from lsm.agents.factory import create_agent
from lsm.agents.models import AgentContext
from lsm.agents.academic import SynthesisAgent
from lsm.agents.phase import PhaseResult
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
            "max_tokens_budget": 18000,
            "max_iterations": 4,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {"synthesis": {"max_iterations": 2}},
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


def _build_agent(tmp_path: Path) -> SynthesisAgent:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    return SynthesisAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_synthesis_agent_runs_and_saves_outputs(tmp_path: Path) -> None:
    source_map_data = json.dumps({"notes/a.md": {"count": 1, "outline": []}})
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("Planning done."),                            # PLAN
        _make_result(                                              # EVIDENCE (with source_map)
            "Evidence gathered.",
            tool_calls=[{"name": "source_map", "result": source_map_data}],
        ),
        _make_result("# Synthesis\n\n- Final point A\n- Final point B\n"),  # SYNTHESIZE
    ]):
        state = agent.run(
            AgentContext(
                messages=[
                    {"role": "user", "content": "Synthesize Aquinas on Eucharistic causality"}
                ]
            )
        )

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.output_path.name == "synthesis.md"
    assert agent.last_result.source_map_path.exists()
    assert agent.last_result.source_map_path.name == "source_map.md"
    assert agent.last_result.log_path.exists()

    synthesis_text = agent.last_result.output_path.read_text(encoding="utf-8")
    source_map_text = agent.last_result.source_map_path.read_text(encoding="utf-8")
    assert "# Synthesis" in synthesis_text
    assert "# Source Map" in source_map_text
    assert "Evidence items:" in source_map_text

    saved_log = agent.last_result.log_path.read_text(encoding="utf-8")
    assert saved_log.strip()


def test_synthesis_agent_output_in_artifacts_dir(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("Planning done."),
        _make_result("Evidence gathered."),
        _make_result("# Synthesis\n\nContent.\n"),
    ]):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert agent.last_result is not None
    assert agent.last_result.output_path.parent.name == "artifacts"
    assert agent.last_result.source_map_path.parent.name == "artifacts"


def test_synthesis_agent_source_map_empty_when_tool_not_called(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("Planning done."),
        _make_result("Evidence gathered."),  # no source_map tool call
        _make_result("# Synthesis\n\nContent.\n"),
    ]):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert agent.last_result is not None
    source_map_text = agent.last_result.source_map_path.read_text(encoding="utf-8")
    assert "# Source Map" in source_map_text
    assert "No source evidence was mapped." in source_map_text


def test_synthesis_agent_phases_run_in_correct_order(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    phase_messages: list[str] = []

    def capture(**kwargs):
        phase_messages.append(kwargs.get("user_message", ""))
        return _make_result("ok")

    with patch.object(BaseAgent, "_run_phase", side_effect=capture):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert len(phase_messages) == 3
    assert "PLAN" in phase_messages[0]
    assert "EVIDENCE" in phase_messages[1]
    assert "SYNTHESIZE" in phase_messages[2]


# ---------------------------------------------------------------------------
# Factory test
# ---------------------------------------------------------------------------

def test_agent_factory_creates_synthesis_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent(
        "synthesis",
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )
    assert isinstance(agent, SynthesisAgent)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result("Planning done."),
        _make_result("Evidence gathered."),
        _make_result("# Synthesis\n\nFactory output\n"),
    ]):
        state = agent.run(
            AgentContext(messages=[{"role": "user", "content": "Factory synthesis topic"}])
        )

    assert state.status.value == "completed"


# ---------------------------------------------------------------------------
# Source-inspection tests
# ---------------------------------------------------------------------------

def test_synthesis_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.academic.synthesis as synthesis_module

    source = inspect.getsource(synthesis_module)
    assert "AgentHarness(" not in source


def test_synthesis_agent_has_no_direct_provider_call() -> None:
    import lsm.agents.academic.synthesis as synthesis_module

    source = inspect.getsource(synthesis_module)
    assert "provider.synthesize(" not in source
    assert "provider.complete(" not in source


def test_synthesis_agent_has_no_direct_sandbox_execute_call() -> None:
    import lsm.agents.academic.synthesis as synthesis_module

    source = inspect.getsource(synthesis_module)
    assert "sandbox.execute(" not in source
