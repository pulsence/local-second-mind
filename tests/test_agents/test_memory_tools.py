from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.memory import Memory, SQLiteMemoryStore
from lsm.agents.tools import (
    MemoryPutTool,
    MemoryRemoveTool,
    MemorySearchTool,
    create_default_tool_registry,
)
from lsm.config.loader import build_config_from_raw
from lsm.config.models.agents import MemoryConfig


def _sqlite_store(tmp_path: Path, name: str = "memory_tools.sqlite3") -> SQLiteMemoryStore:
    cfg = MemoryConfig(storage_backend="sqlite", sqlite_path=tmp_path / name)
    return SQLiteMemoryStore(cfg.sqlite_path, cfg)


def _base_raw(tmp_path: Path, *, memory_enabled: bool = True) -> dict:
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
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
            },
            "memory": {
                "enabled": memory_enabled,
                "storage_backend": "sqlite",
                "sqlite_path": str(tmp_path / "agent-memory.sqlite3"),
            },
        },
    }


def test_memory_put_tool_creates_pending_candidate(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        tool = MemoryPutTool(store)
        payload = json.loads(
            tool.execute(
                {
                    "key": "writing_tone",
                    "value": {"tone": "concise"},
                    "type": "pinned",
                    "scope": "agent",
                    "tags": ["style", "writing"],
                    "rationale": "User consistently requests concise output.",
                    "provenance": "writing_agent",
                }
            )
        )
        assert payload["status"] == "pending"
        assert payload["memory"]["key"] == "writing_tone"
        assert payload["memory"]["type"] == "pinned"
        assert payload["memory"]["scope"] == "agent"

        pending = store.list_candidates(status="pending")
        assert len(pending) == 1
        assert pending[0].id == payload["candidate_id"]
        assert pending[0].provenance == "writing_agent"
        assert pending[0].memory.key == "writing_tone"
    finally:
        store.close()


def test_memory_search_tool_returns_promoted_filtered_records(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        promoted = Memory(
            type="project_fact",
            key="aquinas_eucharist",
            value={"summary": "Instrumental causality"},
            scope="project",
            tags=["aquinas", "theology"],
            source_run_id="run-promoted",
        )
        rejected = Memory(
            type="project_fact",
            key="discarded_note",
            value={"summary": "low confidence"},
            scope="project",
            tags=["aquinas", "theology"],
            source_run_id="run-rejected",
        )
        promoted_id = store.put_candidate(promoted, provenance="research", rationale="stable")
        rejected_id = store.put_candidate(rejected, provenance="research", rationale="weak")
        store.promote(promoted_id)
        store.reject(rejected_id)

        tool = MemorySearchTool(store)
        payload = json.loads(
            tool.execute(
                {
                    "scope": "project",
                    "tags": ["aquinas"],
                    "type": "project_fact",
                    "limit": 10,
                }
            )
        )
        assert len(payload) == 1
        assert payload[0]["key"] == "aquinas_eucharist"
        assert payload[0]["scope"] == "project"
    finally:
        store.close()


def test_memory_put_tool_updates_existing_memory(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        existing = Memory(
            type="project_fact",
            key="writing_style",
            value={"tone": "formal"},
            scope="agent",
            tags=["style"],
            source_run_id="run-initial",
        )
        candidate_id = store.put_candidate(existing, provenance="writer", rationale="initial")
        store.promote(candidate_id)

        tool = MemoryPutTool(store)
        payload = json.loads(
            tool.execute(
                {
                    "memory_id": existing.id,
                    "value": {"tone": "concise"},
                    "tags": ["style", "concise"],
                    "rationale": "Preference refined after feedback.",
                }
            )
        )
        assert payload["operation"] == "update_memory"
        assert payload["status"] == "promoted"
        assert payload["memory_id"] == existing.id
        assert payload["memory"]["value"]["tone"] == "concise"
        assert payload["memory"]["tags"] == ["style", "concise"]

        refreshed = store.get(existing.id)
        assert refreshed.value["tone"] == "concise"
        assert refreshed.tags == ["style", "concise"]
    finally:
        store.close()


def test_memory_remove_tool_deletes_memory(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        existing = Memory(
            type="project_fact",
            key="to_delete",
            value={"keep": False},
            scope="project",
            tags=["cleanup"],
            source_run_id="run-delete",
        )
        candidate_id = store.put_candidate(existing, provenance="curator", rationale="seed")
        store.promote(candidate_id)

        tool = MemoryRemoveTool(store)
        payload = json.loads(tool.execute({"memory_id": existing.id}))
        assert payload["operation"] == "remove_memory"
        assert payload["status"] == "deleted"
        assert payload["memory_id"] == existing.id
    finally:
        store.close()


def test_default_registry_registers_memory_tools_when_memory_enabled(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path, memory_enabled=True), tmp_path / "config.json")
    store = _sqlite_store(tmp_path, "registry-memory.sqlite3")
    try:
        registry = create_default_tool_registry(config, memory_store=store)
        names = {tool.name for tool in registry.list_tools()}
        assert "memory_put" in names
        assert "memory_remove" in names
        assert "memory_search" in names
    finally:
        store.close()


def test_default_registry_skips_memory_tools_when_memory_disabled(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path, memory_enabled=False), tmp_path / "config.json")
    store = _sqlite_store(tmp_path, "registry-disabled.sqlite3")
    try:
        registry = create_default_tool_registry(config, memory_store=store)
        names = {tool.name for tool in registry.list_tools()}
        assert "memory_put" not in names
        assert "memory_remove" not in names
        assert "memory_search" not in names
    finally:
        store.close()
