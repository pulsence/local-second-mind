from __future__ import annotations

import inspect
import json
from pathlib import Path

from lsm.agents.models import AgentContext
from lsm.agents.productivity.librarian import LibrarianAgent
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

    def __init__(self, sources: list[Path]) -> None:
        self.sources = sources

    def execute(self, args: dict) -> str:
        query = str(args.get("query", ""))
        candidates = []
        for idx, source in enumerate(self.sources):
            candidates.append(
                {
                    "id": f"c{idx}",
                    "text": f"Evidence for {query} from {source.name}",
                    "metadata": {"source_path": str(source)},
                    "score": 0.9 - (idx * 0.1),
                }
            )
        return json.dumps({
            "answer": f"Answer for {query}",
            "sources_display": ", ".join(str(s) for s in self.sources),
            "candidates": candidates,
        })


class MemoryPutStubTool(BaseTool):
    name = "memory_put"
    description = "Stub memory tool."
    requires_permission = False
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "object"},
            "type": {"type": "string"},
            "scope": {"type": "string"},
            "tags": {"type": "array"},
            "rationale": {"type": "string"},
        },
        "required": ["key", "value"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, args: dict) -> str:
        self.calls.append(dict(args))
        return json.dumps({"status": "pending", "key": args.get("key")})


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
            "max_tokens_budget": 8000,
            "max_iterations": 3,
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


def test_librarian_agent_builds_idea_graph(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs"
    doc_root.mkdir(parents=True, exist_ok=True)
    doc_a = doc_root / "a.md"
    doc_b = doc_root / "b.md"
    doc_a.write_text("# Alpha\n\nInsight A.\n", encoding="utf-8")
    doc_b.write_text("# Beta\n\nInsight B.\n", encoding="utf-8")

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(QueryKnowledgeBaseStubTool([doc_a, doc_b]))
    memory_tool = MemoryPutStubTool()
    registry.register(memory_tool)
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = LibrarianAgent(config.llm, registry, sandbox, config.agents)

    state = agent.run(
        AgentContext(messages=[{"role": "user", "content": "Make an idea graph"}])
    )
    assert state.status.value == "completed"
    assert agent.last_result is not None
    assert agent.last_result.idea_graph_path.exists()
    assert agent.last_result.graph_json_path.exists()

    idea_graph = agent.last_result.idea_graph_path.read_text(encoding="utf-8")
    assert "## Graph Outline" in idea_graph
    assert str(doc_a) in idea_graph
    assert str(doc_b) in idea_graph

    graph_payload = json.loads(
        agent.last_result.graph_json_path.read_text(encoding="utf-8")
    )
    assert graph_payload["topic"] == "Make an idea graph"
    assert len(graph_payload["sources"]) == 2
    assert memory_tool.calls
    assert memory_tool.calls[0]["key"].startswith("idea_graph:")


def test_librarian_agent_does_not_directly_instantiate_agent_harness() -> None:
    import lsm.agents.productivity.librarian as librarian_module

    source = inspect.getsource(librarian_module)
    assert "AgentHarness(" not in source, (
        "LibrarianAgent must not directly instantiate AgentHarness; use _run_phase() instead"
    )
