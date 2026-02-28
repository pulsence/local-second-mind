from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools import (
    CreateFolderTool,
    QueryKnowledgeBaseTool,
    QueryLLMTool,
    QueryRemoteChainTool,
    QueryRemoteTool,
    ReadFileTool,
    ReadFolderTool,
    ToolRegistry,
    WriteFileTool,
)
from lsm.agents.tools.base import BaseTool
from lsm.config.loader import build_config_from_raw


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args["text"])


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
    }


def test_tool_registry_register_lookup_and_list() -> None:
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.lookup("echo") is tool
    definitions = registry.list_definitions()
    assert definitions[0]["name"] == "echo"
    assert definitions[0]["requires_permission"] is False


def test_file_tools_read_write_create(tmp_path: Path) -> None:
    folder = tmp_path / "out"
    file_path = folder / "note.txt"

    create_tool = CreateFolderTool()
    write_tool = WriteFileTool()
    read_tool = ReadFileTool()

    create_tool.execute({"path": str(folder)})
    result = write_tool.execute({"path": str(file_path), "content": "hello"})
    assert "Wrote 5 chars" in result
    assert read_tool.execute({"path": str(file_path)}) == "hello"


def test_read_folder_tool_lists_entries(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b").mkdir()
    tool = ReadFolderTool()
    output = json.loads(tool.execute({"path": str(tmp_path)}))
    names = {entry["name"] for entry in output}
    assert "a.txt" in names
    assert "b" in names


def test_query_knowledge_base_tool_returns_results(monkeypatch) -> None:
    from lsm.query.api import QueryResult

    class FakeCandidate:
        cid = "c1"
        text = "chunk text"
        distance = 0.2

        @property
        def relevance(self):
            return 1.0 - self.distance

    class FakeConfig:
        pass

    fake_result = QueryResult(
        answer="Test answer",
        sources_display="doc.md",
        candidates=[FakeCandidate()],
        cost=0.0,
        remote_sources=[],
        debug_info={},
    )

    monkeypatch.setattr("lsm.agents.tools.query_knowledge_base.query_sync", lambda **kwargs: fake_result)

    tool = QueryKnowledgeBaseTool(
        config=FakeConfig(),
        embedder=None,
        collection=None,
    )
    payload = json.loads(tool.execute({"query": "test", "top_k": 1}))
    assert "answer" in payload
    assert "candidates" in payload
    assert payload["answer"] == "Test answer"
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["id"] == "c1"


def test_query_llm_tool_uses_provider_factory(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeProvider:
        def send_message(self, **kwargs):
            captured.update(kwargs)
            return "provider-result"

    def fake_create_provider(config):
        return FakeProvider()

    monkeypatch.setattr("lsm.agents.tools.query_llm.create_provider", fake_create_provider)
    monkeypatch.setattr("lsm.agents.tools.query_llm.format_user_content", lambda prompt, context: f"{prompt}|{context}")
    monkeypatch.setattr("lsm.agents.tools.query_llm.get_synthesis_instructions", lambda mode: f"instr:{mode}")

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    tool = QueryLLMTool(config.llm)
    result = tool.execute({"prompt": "hi", "context": "ctx", "mode": "grounded"})
    assert result == "provider-result"
    assert captured["input"] == "hi|ctx"
    assert captured["instruction"] == "instr:grounded"


def test_query_remote_tool_returns_structured_results(monkeypatch, tmp_path: Path) -> None:
    captured_config = {}

    class FakeRemoteProvider:
        def search_structured(self, input_dict, max_results=5):
            return [{"title": "Paper", "url": "https://example.com"}]

    def fake_create_remote_provider(provider_type, config):
        captured_config.update(config)
        return FakeRemoteProvider()

    monkeypatch.setattr(
        "lsm.agents.tools.query_remote.create_remote_provider",
        fake_create_remote_provider,
    )

    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "arxiv", "type": "arxiv", "max_results": 3, "sort_by": "submittedDate"}
    ]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    provider_cfg = config.remote_providers[0]
    tool = QueryRemoteTool(provider_cfg=provider_cfg, config=config)
    assert tool.name == "query_arxiv"
    payload = json.loads(tool.execute({"input": {"query": "ai"}}))
    assert payload[0]["title"] == "Paper"
    assert captured_config["sort_by"] == "submittedDate"


def test_query_remote_chain_tool_executes_chain(monkeypatch, tmp_path: Path) -> None:
    class FakeChain:
        def __init__(self, config, chain_config):
            self.config = config
            self.chain_config = chain_config

        def execute(self, input_dict, max_results=5):
            return [{"title": "Chained"}]

    monkeypatch.setattr("lsm.agents.tools.query_remote_chain.RemoteProviderChain", FakeChain)

    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [
        {"name": "openalex", "type": "openalex"},
        {"name": "crossref", "type": "crossref"},
    ]
    raw["remote_provider_chains"] = [
        {
            "name": "Research Digest",
            "links": [
                {"source": "openalex"},
                {"source": "crossref", "map": ["doi:doi"]},
            ],
        }
    ]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    tool = QueryRemoteChainTool(config)
    payload = json.loads(
        tool.execute({"chain": "Research Digest", "input": {"query": "test"}})
    )
    assert payload[0]["title"] == "Chained"
