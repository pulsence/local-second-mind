from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lsm.agents.base import BaseAgent
from lsm.agents.factory import AgentRegistry, create_agent
from lsm.agents.models import AgentContext
from lsm.agents.academic import ResearchAgent
from lsm.agents.phase import PhaseResult
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class QueryKnowledgeBaseStubTool(BaseTool):
    name = "query_knowledge_base"
    description = "Stub knowledge base query tool."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
            "max_chars": {"type": "integer"},
        },
        "required": ["query"],
    }

    def execute(self, args: dict) -> str:
        query = args.get("query")
        return json.dumps(
            {
                "answer": f"Answer for {query}",
                "sources_display": "docs/source.md",
                "candidates": [
                    {
                        "id": "c1",
                        "text": f"Local snippet for {query}",
                        "score": 0.9,
                        "metadata": {
                            "source_path": "docs/source.md",
                            "title": "Local Source",
                        },
                    }
                ],
            }
        )


def _base_raw(tmp_path: Path) -> dict:
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
            "agent_configs": {"research": {"max_iterations": 2}},
        },
        "remote_providers": [{"name": "arxiv", "type": "arxiv"}],
    }


def _make_result(final_text: str, stop_reason: str = "stop", tool_calls: list | None = None) -> PhaseResult:
    return PhaseResult(
        final_text=final_text,
        tool_calls=tool_calls or [],
        stop_reason=stop_reason,
    )


def _build_agent(tmp_path: Path) -> ResearchAgent:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(QueryKnowledgeBaseStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    return ResearchAgent(
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_research_agent_runs_and_saves_outline(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Scope", "Methods"]'),                          # DECOMPOSE
        _make_result("Scope findings."),                                # RESEARCH:Scope
        _make_result("Methods findings."),                              # RESEARCH:Methods
        _make_result("# Research Outline: AI safety\n\n## Scope\n\nDetails.\n"),  # SYNTHESIZE
        _make_result('{"sufficient": true, "suggestions": []}'),        # REVIEW
    ]):
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "AI safety"}]))

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert "# Research Outline: AI safety" in agent.last_result.outline_markdown
    assert agent.last_result.output_path.exists()
    assert agent.last_result.output_path.parent.name == "artifacts"
    assert agent.last_result.log_path.exists()
    saved_log = agent.last_result.log_path.read_text(encoding="utf-8")
    assert saved_log.strip()


def test_research_agent_phases_execute_in_order(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    phase_user_messages: list[str] = []

    def capture_phase(**kwargs):
        phase_user_messages.append(kwargs.get("user_message", ""))
        if "DECOMPOSE" in kwargs.get("user_message", ""):
            return _make_result('["Scope"]')
        if (kwargs.get("context_label") or "").startswith("subtopic:"):
            return _make_result("Scope findings.")
        if "SYNTHESIZE" in kwargs.get("user_message", ""):
            return _make_result("# Research Outline: Topic\n\n")
        return _make_result('{"sufficient": true, "suggestions": []}')

    with patch.object(BaseAgent, "_run_phase", side_effect=capture_phase):
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert state.status.value == "completed"
    # Verify order: DECOMPOSE → RESEARCH → SYNTHESIZE → REVIEW
    assert "DECOMPOSE" in phase_user_messages[0]
    assert "RESEARCH" in phase_user_messages[1]
    assert "SYNTHESIZE" in phase_user_messages[2]
    assert "REVIEW" in phase_user_messages[3]


def test_research_agent_uses_context_label_per_subtopic(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Scope"]'),                   # DECOMPOSE
        _make_result("Scope findings."),              # RESEARCH:Scope
        _make_result("# Research Outline\n"),         # SYNTHESIZE
        _make_result('{"sufficient": true}'),         # REVIEW
    ]) as mock_phase:
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    calls = mock_phase.call_args_list
    # RESEARCH call (index 1) must have context_label="subtopic:Scope"
    assert calls[1].kwargs.get("context_label") == "subtopic:Scope"
    # SYNTHESIZE call (index 2) must have context_label=None
    synthesize_kwargs = calls[2].kwargs
    assert "context_label" in synthesize_kwargs
    assert synthesize_kwargs["context_label"] is None


def test_research_agent_subtopic_names_in_logs(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Scope", "Methods"]'),
        _make_result("Scope findings."),
        _make_result("Methods findings."),
        _make_result("# Research Outline\n"),
        _make_result('{"sufficient": true}'),
    ]):
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    log_messages = [entry.content for entry in state.log_entries]
    # Iteration log should mention both subtopic names
    assert any("Scope" in msg and "Methods" in msg for msg in log_messages)
    # Per-subtopic collecting logs
    assert any("Collecting findings for subtopic: Scope" in msg for msg in log_messages)
    assert any("Collecting findings for subtopic: Methods" in msg for msg in log_messages)


def test_research_agent_passes_findings_to_synthesize(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    captured_synthesize_msg: list[str] = []

    def capture(**kwargs):
        msg = kwargs.get("user_message", "")
        if "SYNTHESIZE" in msg:
            captured_synthesize_msg.append(msg)
            return _make_result("# Research Outline\n")
        if "DECOMPOSE" in msg:
            return _make_result('["Scope"]')
        if (kwargs.get("context_label") or "").startswith("subtopic:"):
            return _make_result("Finding: Key insight [S1].")
        return _make_result('{"sufficient": true}')

    with patch.object(BaseAgent, "_run_phase", side_effect=capture):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert captured_synthesize_msg
    assert "Finding: Key insight [S1]." in captured_synthesize_msg[0]


def test_research_agent_stops_when_budget_exhausted(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Sub1", "Sub2", "Sub3"]'),                        # DECOMPOSE (3 subtopics)
        _make_result("Sub1 findings.", stop_reason="budget_exhausted"),  # RESEARCH:Sub1 → breaks
        # Sub2 and Sub3 RESEARCH must NOT be called
        _make_result("# Outline\n"),                                     # SYNTHESIZE
        _make_result('{"sufficient": true, "suggestions": []}'),         # REVIEW
    ]) as mock_phase:
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Big topic"}]))

    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    # Only 4 calls (DECOMPOSE + Sub1 RESEARCH + SYNTHESIZE + REVIEW), not 6
    assert mock_phase.call_count == 4


def test_research_agent_review_suggestions_logged(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Scope"]'),
        _make_result("Scope findings."),
        _make_result("# Research Outline\n"),
        _make_result('{"sufficient": false, "suggestions": ["Add examples", "Expand coverage"]}'),
    ]):
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    log_messages = [entry.content for entry in state.log_entries]
    assert any("Refining with 2 review suggestions" in msg for msg in log_messages)
    assert any("Add examples" in msg for msg in log_messages)


def test_research_agent_output_in_artifacts_dir(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["Scope"]'),
        _make_result("Scope findings."),
        _make_result("# Research Outline: Topic\n"),
        _make_result('{"sufficient": true}'),
    ]):
        agent.run(AgentContext(messages=[{"role": "user", "content": "Topic"}]))

    assert agent.last_result is not None
    output_path = agent.last_result.output_path
    assert output_path.exists()
    assert output_path.parent.name == "artifacts"
    assert output_path.suffix == ".md"


# ---------------------------------------------------------------------------
# Factory and registry tests
# ---------------------------------------------------------------------------

def test_agent_factory_creates_research_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(QueryKnowledgeBaseStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent(
        "research",
        config.llm,
        registry,
        sandbox,
        config.agents,
        lsm_config=config,
    )
    assert isinstance(agent, ResearchAgent)

    with patch.object(BaseAgent, "_run_phase", side_effect=[
        _make_result('["S1"]'),
        _make_result("S1 findings."),
        _make_result("# Research Outline: Factory topic\n"),
        _make_result('{"sufficient": true, "suggestions": []}'),
    ]):
        state = agent.run(AgentContext(messages=[{"role": "user", "content": "Factory topic"}]))

    assert state.status.value == "completed"


def test_agent_registry_rejects_unknown_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    assert config.agents is not None
    registry = ToolRegistry()
    registry.register(QueryKnowledgeBaseStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent_registry = AgentRegistry()

    with pytest.raises(ValueError, match="Unknown agent"):
        agent_registry.create(
            "unknown",
            llm_registry=config.llm,
            tool_registry=registry,
            sandbox=sandbox,
            agent_config=config.agents,
            lsm_config=config,
        )


# ---------------------------------------------------------------------------
# Source-inspection tests (no direct harness/provider/sandbox calls)
# ---------------------------------------------------------------------------

def test_research_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.academic.research as research_module

    source = inspect.getsource(research_module)
    assert "AgentHarness(" not in source, (
        "ResearchAgent must not directly instantiate AgentHarness; use _run_phase() instead"
    )


def test_research_agent_has_no_direct_provider_synthesize_call() -> None:
    import lsm.agents.academic.research as research_module

    source = inspect.getsource(research_module)
    assert "provider.synthesize(" not in source, (
        "ResearchAgent must not call provider.synthesize() directly"
    )


def test_research_agent_has_no_direct_sandbox_execute_call() -> None:
    import lsm.agents.academic.research as research_module

    source = inspect.getsource(research_module)
    assert "sandbox.execute(" not in source, (
        "ResearchAgent must not call sandbox.execute() directly"
    )
