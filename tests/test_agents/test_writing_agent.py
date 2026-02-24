from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.factory import create_agent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.agents.productivity import WritingAgent
from lsm.config.loader import build_config_from_raw


class QueryEmbeddingsStubTool(BaseTool):
    name = "query_embeddings"
    description = "Stub local retrieval tool."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    }

    def execute(self, args: dict) -> str:
        query = str(args.get("query", ""))
        return json.dumps(
            [
                {
                    "id": "c1",
                    "text": f"evidence for {query}",
                    "metadata": {"source_path": "notes/a.md"},
                    "relevance": 0.88,
                },
                {
                    "id": "c2",
                    "text": f"secondary evidence for {query}",
                    "metadata": {"source_path": "notes/b.md"},
                    "relevance": 0.81,
                },
            ]
        )


class ExtractSnippetsStubTool(BaseTool):
    name = "extract_snippets"
    description = "Stub snippet extraction tool."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "paths": {"type": "array"},
            "max_snippets": {"type": "integer"},
        },
        "required": ["query", "paths"],
    }

    def execute(self, args: dict) -> str:
        query = str(args.get("query", ""))
        paths = args.get("paths") or []
        snippets = []
        for path in paths:
            snippets.append(
                {
                    "source_path": str(path),
                    "snippet": f"snippet for {query} from {path}",
                    "score": 0.77,
                }
            )
        return json.dumps(snippets)


class SourceMapStubTool(BaseTool):
    name = "source_map"
    description = "Stub source map aggregator."
    input_schema = {
        "type": "object",
        "properties": {"evidence": {"type": "array"}},
        "required": ["evidence"],
    }

    def execute(self, args: dict) -> str:
        evidence = args.get("evidence") or []
        output: dict[str, dict[str, object]] = {}
        for item in evidence:
            path = str(item.get("source_path", "unknown"))
            output.setdefault(path, {"count": 0, "top_snippets": []})
            output[path]["count"] = int(output[path]["count"]) + 1
            snippet = str(item.get("snippet", "")).strip()
            if snippet and snippet not in output[path]["top_snippets"]:
                output[path]["top_snippets"].append(snippet)
        return json.dumps(output)


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
            "agent_configs": {"writing": {"max_iterations": 2}},
        },
    }


def test_writing_agent_runs_and_saves_deliverable(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = context, mode, kwargs
            lower = str(question).lower()
            if "concise markdown outline" in lower:
                return "# Outline\n\n## Intro\n\n## Argument\n\n## Conclusion\n"
            if "draft a polished markdown deliverable" in lower:
                return "# Draft Deliverable\n\nInitial grounded draft.\n"
            if "revise the draft for clarity" in lower:
                return "# Final Deliverable\n\nRevised grounded draft.\n"
            return "# Final Deliverable\n\nFallback.\n"

    monkeypatch.setattr(
        "lsm.agents.productivity.writing.create_provider",
        lambda cfg: FakeProvider(),
    )

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    registry.register(ExtractSnippetsStubTool())
    registry.register(SourceMapStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = WritingAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Write on sacramental realism"}]))
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.log_path.exists()
    content = agent.last_result.output_path.read_text(encoding="utf-8")
    assert "# Final Deliverable" in content

    saved_log = agent.last_result.log_path.read_text(encoding="utf-8")
    assert saved_log.strip()


def test_agent_factory_creates_writing_agent(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return "# deliverable"

    monkeypatch.setattr(
        "lsm.agents.productivity.writing.create_provider",
        lambda cfg: FakeProvider(),
    )

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryEmbeddingsStubTool())
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent("writing", config.llm, registry, sandbox, config.agents)
    assert isinstance(agent, WritingAgent)
    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Factory writing topic"}]))
    assert state.status.value == "completed"
