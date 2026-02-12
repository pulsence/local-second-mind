from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.curator import CuratorAgent
from lsm.agents.factory import create_agent
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class ReadFolderStubTool(BaseTool):
    name = "read_folder"
    description = "Stub folder listing tool."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        root = Path(str(args.get("path", ".")).strip())
        return json.dumps(
            [
                {"name": "a.md", "path": str(root / "notes" / "a.md"), "is_dir": False},
                {"name": "b.md", "path": str(root / "notes" / "b.md"), "is_dir": False},
                {"name": "c.md", "path": str(root / "notes" / "c.md"), "is_dir": False},
                {"name": "folder", "path": str(root / "notes" / "folder"), "is_dir": True},
            ]
        )


class FileMetadataStubTool(BaseTool):
    name = "file_metadata"
    description = "Stub file metadata tool."
    input_schema = {
        "type": "object",
        "properties": {"paths": {"type": "array"}},
        "required": ["paths"],
    }

    def execute(self, args: dict) -> str:
        paths = [str(item) for item in (args.get("paths") or [])]
        payload = []
        for path in paths:
            if path.endswith("a.md"):
                payload.append(
                    {
                        "path": path,
                        "size_bytes": 100,
                        "mtime_iso": "2022-01-01T00:00:00+00:00",
                        "ext": ".md",
                    }
                )
            elif path.endswith("b.md"):
                payload.append(
                    {
                        "path": path,
                        "size_bytes": 0,
                        "mtime_iso": "2025-01-01T00:00:00+00:00",
                        "ext": ".md",
                    }
                )
            else:
                payload.append(
                    {
                        "path": path,
                        "size_bytes": 900,
                        "mtime_iso": "2026-01-01T00:00:00+00:00",
                        "ext": ".md",
                    }
                )
        return json.dumps(payload)


class HashFileStubTool(BaseTool):
    name = "hash_file"
    description = "Stub hash tool."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        path = str(args.get("path", ""))
        if path.endswith("a.md") or path.endswith("b.md"):
            digest = "dup-hash"
        else:
            digest = "unique-hash"
        return json.dumps({"path": path, "sha256": digest})


class SimilaritySearchStubTool(BaseTool):
    name = "similarity_search"
    description = "Stub similarity search tool."
    input_schema = {
        "type": "object",
        "properties": {
            "paths": {"type": "array"},
            "top_k": {"type": "integer"},
            "threshold": {"type": "number"},
        },
        "required": [],
    }

    def execute(self, args: dict) -> str:
        paths = args.get("paths") or []
        if len(paths) < 2:
            return "[]"
        return json.dumps(
            [
                {
                    "source_path_a": str(paths[0]),
                    "source_path_b": str(paths[2] if len(paths) > 2 else paths[1]),
                    "similarity": 0.94,
                }
            ]
        )


class QueryEmbeddingsStubTool(BaseTool):
    name = "query_embeddings"
    description = "Stub retrieval tool."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
        "required": ["query"],
    }

    def execute(self, args: dict) -> str:
        _ = args
        return json.dumps(
            [
                {
                    "id": "q1",
                    "text": "TODO: expand this draft.",
                    "relevance": 0.8,
                    "metadata": {"source_path": "notes/a.md"},
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
            "max_tokens_budget": 20000,
            "max_iterations": 4,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {
                "curator": {
                    "max_iterations": 2,
                    "stale_days": 365,
                    "near_duplicate_threshold": 0.9,
                }
            },
        },
    }


def _register_tools(registry: ToolRegistry) -> None:
    registry.register(ReadFolderStubTool())
    registry.register(FileMetadataStubTool())
    registry.register(HashFileStubTool())
    registry.register(SimilaritySearchStubTool())
    registry.register(QueryEmbeddingsStubTool())


def test_curator_agent_runs_and_saves_report(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = context, mode, kwargs
            lower = str(question).lower()
            if "select corpus curation scope" in lower:
                return json.dumps(
                    {
                        "scope_path": str(tmp_path),
                        "stale_days": 365,
                        "near_duplicate_threshold": 0.9,
                        "top_near_duplicates": 10,
                    }
                )
            if "produce concise actionable corpus curation recommendations" in lower:
                return json.dumps(
                    [
                        "Consolidate exact duplicates.",
                        "Review near-duplicate pairs for merges.",
                    ]
                )
            return "[]"

    monkeypatch.setattr("lsm.agents.curator.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    _register_tools(registry)
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = CuratorAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Curate theology notes"}])
    )
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.output_path.exists()
    assert agent.last_result.output_path.name == "curation_report.md"
    assert agent.last_result.log_path.exists()

    report = agent.last_result.output_path.read_text(encoding="utf-8")
    assert "# Curation Report: Curate theology notes" in report
    assert "## Exact Duplicates" in report
    assert "## Near Duplicates" in report
    assert "## Heuristics" in report
    assert "Consolidate exact duplicates." in report

    saved_log = json.loads(agent.last_result.log_path.read_text(encoding="utf-8"))
    assert isinstance(saved_log, list)
    assert len(saved_log) > 0


def test_agent_factory_creates_curator_agent(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = context, mode, kwargs
            lower = str(question).lower()
            if "select corpus curation scope" in lower:
                return json.dumps(
                    {
                        "scope_path": str(tmp_path),
                        "stale_days": 365,
                        "near_duplicate_threshold": 0.9,
                        "top_near_duplicates": 8,
                    }
                )
            if "produce concise actionable corpus curation recommendations" in lower:
                return json.dumps(["No urgent quality issues detected."])
            return "[]"

    monkeypatch.setattr("lsm.agents.curator.create_provider", lambda cfg: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    _register_tools(registry)
    sandbox = ToolSandbox(config.agents.sandbox)

    agent = create_agent("curator", config.llm, registry, sandbox, config.agents)
    assert isinstance(agent, CuratorAgent)
    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Factory curator topic"}])
    )
    assert state.status.value == "completed"
