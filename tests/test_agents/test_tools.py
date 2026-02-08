from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.tools import (
    CreateFolderTool,
    QueryEmbeddingsTool,
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
from lsm.vectordb.base import VectorDBQueryResult


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


def test_query_embeddings_tool_returns_hits() -> None:
    class FakeEmbedder:
        def encode(self, texts, **kwargs):
            return [[0.1, 0.2, 0.3]]

    class FakeCollection:
        def query(self, embedding, top_k, filters=None):
            return VectorDBQueryResult(
                ids=["c1"],
                documents=["chunk text"],
                metadatas=[{"source_path": "doc.md"}],
                distances=[0.2],
            )

    tool = QueryEmbeddingsTool(
        collection=FakeCollection(),
        embedder=FakeEmbedder(),
        batch_size=8,
    )
    payload = json.loads(tool.execute({"query": "test", "top_k": 1}))
    assert len(payload) == 1
    assert payload[0]["id"] == "c1"
    assert payload[0]["metadata"]["source_path"] == "doc.md"


def test_query_llm_tool_uses_provider_factory(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        def synthesize(self, question, context, mode="insight", **kwargs):
            return f"mode={mode}|{question}|{context}"

    def fake_create_provider(config):
        return FakeProvider()

    monkeypatch.setattr("lsm.agents.tools.query_llm.create_provider", fake_create_provider)

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    tool = QueryLLMTool(config.llm)
    result = tool.execute({"prompt": "hi", "context": "ctx", "mode": "grounded"})
    assert result == "mode=grounded|hi|ctx"


def test_query_remote_tool_returns_structured_results(monkeypatch, tmp_path: Path) -> None:
    class FakeRemoteProvider:
        def search_structured(self, input_dict, max_results=5):
            return [{"title": "Paper", "url": "https://example.com"}]

    def fake_create_remote_provider(provider_type, config):
        return FakeRemoteProvider()

    monkeypatch.setattr(
        "lsm.agents.tools.query_remote.create_remote_provider",
        fake_create_remote_provider,
    )

    raw = _base_raw(tmp_path)
    raw["remote_providers"] = [{"name": "arxiv", "type": "arxiv", "max_results": 3}]
    config = build_config_from_raw(raw, tmp_path / "config.json")
    tool = QueryRemoteTool(config)
    payload = json.loads(tool.execute({"provider": "arxiv", "input": {"query": "ai"}}))
    assert payload[0]["title"] == "Paper"


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

