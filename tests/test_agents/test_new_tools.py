from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.agents.tools import (
    AppendFileTool,
    ExtractSnippetsTool,
    FileMetadataTool,
    HashFileTool,
    SimilaritySearchTool,
    SourceMapTool,
    create_default_tool_registry,
)
from lsm.config.loader import build_config_from_raw
from lsm.vectordb.base import VectorDBGetResult, VectorDBQueryResult


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


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        _ = texts, kwargs
        return [[0.5, 0.5]]


class FakeVectorCollection:
    def __init__(self) -> None:
        self.query_calls: List[Optional[Dict[str, Any]]] = []
        self.path_call_count = 0

    def query(self, embedding, top_k, filters=None):
        _ = embedding, top_k
        self.query_calls.append(filters)
        source_path = None
        if isinstance(filters, dict):
            source_path = filters.get("source_path")

        if source_path == "notes/a.md":
            return VectorDBQueryResult(
                ids=["a1"],
                documents=["Alpha snippet from A"],
                metadatas=[{"source_path": "notes/a.md"}],
                distances=[0.1],
            )
        if source_path == "notes/b.md":
            return VectorDBQueryResult(
                ids=["b1"],
                documents=["Beta snippet from B"],
                metadatas=[{"source_path": "notes/b.md"}],
                distances=[0.3],
            )
        return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

    def get(self, ids=None, filters=None, limit=None, offset=0, include=None):
        _ = limit, offset, include
        if ids is not None:
            records = {
                "c1": {"source_path": "notes/a.md", "embedding": [1.0, 0.0]},
                "c2": {"source_path": "notes/b.md", "embedding": [0.9, 0.1]},
                "c3": {"source_path": "notes/c.md", "embedding": [-1.0, 0.0]},
            }
            selected = [cid for cid in ids if cid in records]
            return VectorDBGetResult(
                ids=selected,
                metadatas=[
                    {"source_path": records[cid]["source_path"]}
                    for cid in selected
                ],
                embeddings=[records[cid]["embedding"] for cid in selected],
            )

        source_path = None
        if isinstance(filters, dict):
            source_path = filters.get("source_path")
        if source_path == "notes/a.md":
            self.path_call_count += 1
            return VectorDBGetResult(
                ids=["p1"],
                metadatas=[{"source_path": "notes/a.md"}],
                embeddings=[[1.0, 0.0]],
            )
        if source_path == "notes/b.md":
            self.path_call_count += 1
            return VectorDBGetResult(
                ids=["p2"],
                metadatas=[{"source_path": "notes/b.md"}],
                embeddings=[[0.95, 0.05]],
            )
        return VectorDBGetResult(ids=[], metadatas=[], embeddings=[])


def test_extract_snippets_scopes_to_paths_and_limits_results() -> None:
    tool = ExtractSnippetsTool(
        collection=FakeVectorCollection(),
        embedder=FakeEmbedder(),
        batch_size=8,
    )
    payload = json.loads(
        tool.execute(
            {
                "query": "alpha",
                "paths": ["notes/a.md", "notes/b.md"],
                "max_snippets": 1,
                "max_chars_per_snippet": 8,
            }
        )
    )
    assert len(payload) == 1
    assert payload[0]["source_path"] == "notes/a.md"
    assert payload[0]["snippet"] == "Alpha sn"
    assert payload[0]["score"] > 0.0


def test_file_metadata_returns_size_mtime_and_extension(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("hello", encoding="utf-8")
    tool = FileMetadataTool()
    payload = json.loads(tool.execute({"paths": [str(path)]}))
    assert payload[0]["path"] == str(path.resolve())
    assert payload[0]["size_bytes"] == 5
    assert payload[0]["ext"] == ".txt"
    assert "T" in payload[0]["mtime_iso"]


def test_hash_file_returns_sha256(tmp_path: Path) -> None:
    path = tmp_path / "hash.txt"
    path.write_text("hello", encoding="utf-8")
    tool = HashFileTool()
    payload = json.loads(tool.execute({"path": str(path)}))
    assert payload["path"] == str(path.resolve())
    assert payload["sha256"] == hashlib.sha256(b"hello").hexdigest()


def test_similarity_search_finds_top_pairs_for_chunk_ids() -> None:
    tool = SimilaritySearchTool(collection=FakeVectorCollection())
    payload = json.loads(
        tool.execute(
            {
                "chunk_ids": ["c1", "c2", "c3"],
                "top_k": 5,
                "threshold": 0.8,
            }
        )
    )
    assert len(payload) == 1
    assert payload[0]["id_a"] == "c1"
    assert payload[0]["id_b"] == "c2"
    assert payload[0]["similarity"] >= 0.8


def test_similarity_search_supports_path_input() -> None:
    collection = FakeVectorCollection()
    tool = SimilaritySearchTool(collection=collection)
    payload = json.loads(
        tool.execute(
            {
                "paths": ["notes/a.md", "notes/b.md"],
                "top_k": 3,
                "threshold": 0.9,
            }
        )
    )
    assert len(payload) == 1
    assert payload[0]["source_path_a"] == "notes/a.md"
    assert payload[0]["source_path_b"] == "notes/b.md"
    assert collection.path_call_count == 2


def test_source_map_aggregates_counts_and_outlines(tmp_path: Path) -> None:
    path_a = tmp_path / "a.md"
    path_b = tmp_path / "b.md"
    path_a.write_text("# A\n\nAlpha.", encoding="utf-8")
    path_b.write_text("# B\n\nBeta.", encoding="utf-8")

    tool = SourceMapTool()
    payload = json.loads(
        tool.execute(
            {
                "evidence": [
                    {"source_path": str(path_a), "snippet": "A1", "score": 0.9},
                    {"source_path": str(path_a), "snippet": "A2", "score": 0.4},
                    {"source_path": str(path_a), "snippet": "A1", "score": 0.3},
                    {"source_path": str(path_b), "snippet": "B1", "score": 0.8},
                ],
                "max_depth": 1,
            }
        )
    )
    assert payload[str(path_a)]["count"] == 3
    assert payload[str(path_b)]["count"] == 1
    assert payload[str(path_a)]["outline"]
    assert payload[str(path_b)]["outline"]


def test_append_file_appends_content(tmp_path: Path) -> None:
    path = tmp_path / "append.txt"
    path.write_text("start", encoding="utf-8")
    tool = AppendFileTool()
    result = tool.execute({"path": str(path), "content": "-end"})
    assert "Appended 4 chars" in result
    assert path.read_text(encoding="utf-8") == "start-end"


def test_default_tool_registry_registers_new_tools_with_dependency_rules(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")

    no_vectors = create_default_tool_registry(config)
    no_vectors_names = {tool.name for tool in no_vectors.list_tools()}
    assert "find_file" in no_vectors_names
    assert "find_section" in no_vectors_names
    assert "edit_file" in no_vectors_names
    assert "file_metadata" in no_vectors_names
    assert "hash_file" in no_vectors_names
    assert "source_map" in no_vectors_names
    assert "append_file" in no_vectors_names
    assert "ask_user" in no_vectors_names
    assert "spawn_agent" in no_vectors_names
    assert "await_agent" in no_vectors_names
    assert "collect_artifacts" in no_vectors_names
    assert "bash" in no_vectors_names
    assert "powershell" in no_vectors_names
    assert "similarity_search" not in no_vectors_names
    assert "extract_snippets" not in no_vectors_names

    with_vectors = create_default_tool_registry(
        config,
        collection=FakeVectorCollection(),
        embedder=FakeEmbedder(),
        batch_size=4,
    )
    with_vectors_names = {tool.name for tool in with_vectors.list_tools()}
    assert "similarity_search" in with_vectors_names
    assert "query_knowledge_base" in with_vectors_names
    assert "extract_snippets" in with_vectors_names
    assert "ask_user" in with_vectors_names
    assert "spawn_agent" in with_vectors_names
    assert "await_agent" in with_vectors_names
    assert "collect_artifacts" in with_vectors_names
    assert "bash" in with_vectors_names
    assert "powershell" in with_vectors_names
