from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.factory import create_agent
from lsm.agents.models import AgentContext
from lsm.agents.synthesis import SynthesisAgent
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class ReadFolderStubTool(BaseTool):
    name = "read_folder"
    description = "Stub folder listing tool."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean"},
        },
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        root = Path(str(args.get("path", ".")).strip())
        return json.dumps(
            [
                {
                    "name": "a.md",
                    "path": str(root / "notes" / "a.md"),
                    "is_dir": False,
                },
                {
                    "name": "b.md",
                    "path": str(root / "notes" / "b.md"),
                    "is_dir": False,
                },
            ]
        )


class QueryEmbeddingsStubTool(BaseTool):
    name = "query_embeddings"
    description = "Stub local retrieval tool."
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
        query = str(args.get("query", ""))
        return json.dumps(
            [
                {
                    "id": "hit-1",
                    "text": f"Primary evidence for {query}",
                    "metadata": {"source_path": "notes/a.md"},
                    "relevance": 0.89,
                },
                {
                    "id": "hit-2",
                    "text": f"Secondary evidence for {query}",
                    "metadata": {"source_path": "notes/b.md"},
                    "relevance": 0.81,
                },
            ]
        )


class ExtractSnippetsStubTool(BaseTool):
    name = "extract_snippets"
    description = "Stub snippet extraction."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "paths": {"type": "array"},
            "max_snippets": {"type": "integer"},
            "max_chars_per_snippet": {"type": "integer"},
        },
        "required": ["query", "paths"],
    }

    def execute(self, args: dict) -> str:
        query = str(args.get("query", ""))
        paths = args.get("paths") or []
        payload = []
        for path in paths:
            payload.append(
                {
                    "source_path": str(path),
                    "snippet": f"Snippet for {query} from {path}",
                    "score": 0.75,
                }
            )
        return json.dumps(payload)


class SourceMapStubTool(BaseTool):
    name = "source_map"
    description = "Stub source map tool."
    input_schema = {
        "type": "object",
        "properties": {
            "evidence": {"type": "array"},
            "max_snippets_per_source": {"type": "integer"},
        },
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


class ReadFileStubTool(BaseTool):
    name = "read_file"
    description = "Stub read file tool."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        _ = args
        return "fallback file snippet"


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


def _register_tools(registry: ToolRegistry) -> None:
    registry.register(ReadFolderStubTool())
    registry.register(QueryEmbeddingsStubTool())
    registry.register(ExtractSnippetsStubTool())
    registry.register(SourceMapStubTool())
    registry.register(ReadFileStubTool())


def test_synthesis_agent_runs_and_saves_outputs(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = context, mode, kwargs
            lower = str(question).lower()
            if "select synthesis scope and output settings" in lower:
                return json.dumps(
                    {
                        "query": "Eucharistic causality",
                        "target_format": "outline",
                        "target_length_words": 220,
                    }
                )
            if "synthesize grounded output in markdown" in lower:
                return "# Synthesis Draft\n\n- Point A\n- Point B\n"
            if "tighten the synthesis" in lower:
                return "# Synthesis\n\n- Final point A\n- Final point B\n"
            if "assess whether the synthesis covers the core evidence" in lower:
                return json.dumps(
                    {
                        "sufficient": True,
                        "coverage_notes": [],
                        "missing_topics": [],
                    }
                )
            return "# Synthesis\n\nFallback\n"

    monkeypatch.setattr("lsm.agents.synthesis.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    _register_tools(registry)
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = SynthesisAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(
            messages=[
                {
                    "role": "user",
                    "content": "Synthesize Aquinas on Eucharistic causality",
                }
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


def test_agent_factory_creates_synthesis_agent(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = context, mode, kwargs
            lower = str(question).lower()
            if "select synthesis scope and output settings" in lower:
                return json.dumps(
                    {
                        "query": "Factory synthesis",
                        "target_format": "bullets",
                        "target_length_words": 180,
                    }
                )
            if "assess whether the synthesis covers the core evidence" in lower:
                return json.dumps({"sufficient": True, "coverage_notes": [], "missing_topics": []})
            return "# Synthesis\n\nFactory output\n"

    monkeypatch.setattr("lsm.agents.synthesis.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    _register_tools(registry)
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent("synthesis", config.llm, registry, sandbox, config.agents)
    assert isinstance(agent, SynthesisAgent)
    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Factory synthesis topic"}])
    )
    assert state.status.value == "completed"
