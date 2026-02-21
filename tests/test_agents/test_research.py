from __future__ import annotations

import json
from pathlib import Path

import pytest

from lsm.agents.factory import AgentRegistry, create_agent
from lsm.agents.models import AgentContext
from lsm.agents.research import ResearchAgent
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class QueryEmbeddingsStubTool(BaseTool):
    name = "query_embeddings"
    description = "Stub local retrieval tool."
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}}

    def execute(self, args: dict) -> str:
        return json.dumps([{"source": "local", "query": args.get("query"), "score": 0.9}])


class QueryRemoteStubTool(BaseTool):
    name = "query_remote"
    description = "Stub remote retrieval tool."
    input_schema = {
        "type": "object",
        "properties": {"provider": {"type": "string"}, "input": {"type": "object"}},
    }

    def execute(self, args: dict) -> str:
        return json.dumps(
            [
                {
                    "source": "remote",
                    "provider": args.get("provider"),
                    "query": (args.get("input") or {}).get("query"),
                    "score": 0.8,
                }
            ]
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


def test_research_agent_runs_and_saves_outline(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            lower = str(question).lower()
            if "decompose the topic" in lower:
                return json.dumps(["Scope", "Methods"])
            if "select the best tools" in lower:
                return json.dumps({"tools": ["query_embeddings", "query_remote"]})
            if "review this research outline" in lower:
                return json.dumps({"sufficient": True, "suggestions": []})
            if "summarize findings for subtopic" in lower:
                return "- Key finding\n- Supporting source"
            return json.dumps({"sufficient": True, "suggestions": []})

    monkeypatch.setattr("lsm.agents.research.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    registry.register(QueryRemoteStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = ResearchAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "AI safety"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert "# Research Outline: AI safety" in agent.last_result.outline_markdown
    assert agent.last_result.output_path.exists()
    assert agent.last_result.log_path.exists()
    saved_log = agent.last_result.log_path.read_text(encoding="utf-8")
    assert saved_log.strip()


def test_research_agent_stops_when_budget_exhausted(monkeypatch, tmp_path: Path) -> None:
    class VerboseProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            lower = str(question).lower()
            if "decompose the topic" in lower:
                return json.dumps(["Large Topic"])
            if "select the best tools" in lower:
                return json.dumps({"tools": ["query_embeddings"]})
            if "review this research outline" in lower:
                return json.dumps({"sufficient": False, "suggestions": ["Refine"]})
            return "x" * 5000

    monkeypatch.setattr("lsm.agents.research.create_provider", lambda cfg: VerboseProvider())

    raw = _base_raw(tmp_path)
    raw["agents"]["max_tokens_budget"] = 200
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = ResearchAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Big topic"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()


def test_agent_factory_creates_research_agent(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            lower = str(question).lower()
            if "decompose the topic" in lower:
                return json.dumps(["S1"])
            if "select the best tools" in lower:
                return json.dumps({"tools": ["query_embeddings"]})
            if "review this research outline" in lower:
                return json.dumps({"sufficient": True, "suggestions": []})
            return "- Summary"

    monkeypatch.setattr("lsm.agents.research.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent("research", config.llm, registry, sandbox, config.agents)
    assert isinstance(agent, ResearchAgent)
    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Factory topic"}]))
    assert state.status.value == "completed"


def test_agent_registry_rejects_unknown_agent(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent_registry = AgentRegistry()

    with pytest.raises(ValueError, match="Unknown agent"):
        agent_registry.create(
            "unknown",
            llm_registry=config.llm,
            tool_registry=registry,
            sandbox=sandbox,
            agent_config=config.agents,
        )
