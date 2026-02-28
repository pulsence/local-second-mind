from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.memory import Memory, MemoryContextBuilder, SQLiteMemoryStore
from lsm.agents.models import AgentContext
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw
from lsm.config.models.agents import MemoryConfig
from lsm.agents.memory.models import now_utc


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("text", ""))


def _sqlite_store(tmp_path: Path, name: str = "memory_context.sqlite3") -> SQLiteMemoryStore:
    cfg = MemoryConfig(storage_backend="sqlite")
    conn = sqlite3.connect(str(tmp_path / name))
    return SQLiteMemoryStore(conn, cfg, owns_connection=True)


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
            "provider": "sqlite",
            "path": str(tmp_path / "data"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 5000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "memory": {
                "enabled": True,
                "storage_backend": "sqlite",
            },
            "agent_configs": {},
        },
    }


def test_memory_context_builder_builds_block_and_marks_only_injected(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        base_time = now_utc() - timedelta(days=15)
        agent_mem = Memory(
            type="task_state",
            key="agent_constraint",
            value={"constraint": "avoid jargon"},
            scope="agent",
            tags=["aquinas"],
            last_used_at=base_time,
            source_run_id="run-agent",
        )
        project_mem = Memory(
            type="project_fact",
            key="project_fact",
            value={"fact": "uses instrumental causality"},
            scope="project",
            tags=["aquinas"],
            last_used_at=base_time,
            source_run_id="run-project",
        )
        global_mem = Memory(
            type="pinned",
            key="global_style",
            value={"style": "concise"},
            scope="global",
            tags=["aquinas"],
            last_used_at=base_time,
            source_run_id="run-global",
        )
        untouched_mem = Memory(
            type="project_fact",
            key="other_topic",
            value={"fact": "unrelated"},
            scope="project",
            tags=["unrelated"],
            last_used_at=base_time,
            source_run_id="run-other",
        )

        promoted_ids: list[str] = []
        for memory in (agent_mem, project_mem, global_mem, untouched_mem):
            candidate_id = store.put_candidate(memory, provenance="test", rationale="seed")
            store.promote(candidate_id)
            promoted_ids.append(memory.id)

        before_selected = {
            agent_mem.id: store.get(agent_mem.id).last_used_at,
            project_mem.id: store.get(project_mem.id).last_used_at,
            global_mem.id: store.get(global_mem.id).last_used_at,
        }
        before_untouched = store.get(untouched_mem.id).last_used_at

        builder = MemoryContextBuilder(store, default_limit=3, default_token_budget=800)
        payload = builder.build_payload(
            agent_name="research",
            topic="How does Aquinas understand the Eucharist?",
        )

        assert payload.text
        assert "Standing Context (Memory)" in payload.text
        assert "agent_constraint" in payload.text
        assert "project_fact" in payload.text
        assert "global_style" in payload.text
        assert len(payload.memories) == 3

        for memory_id, previous in before_selected.items():
            assert store.get(memory_id).last_used_at >= previous
        assert store.get(untouched_mem.id).last_used_at == before_untouched
    finally:
        store.close()


def test_memory_context_builder_allows_empty_result(tmp_path: Path) -> None:
    store = _sqlite_store(tmp_path)
    try:
        builder = MemoryContextBuilder(store)
        payload = builder.build_payload(agent_name="research", topic="Anything")
        assert payload.text == ""
        assert payload.memories == []
    finally:
        store.close()


def test_harness_injects_memory_standing_context(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.last_system_prompt = ""
            self.last_user_prompt = ""

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = temperature, max_tokens, kwargs
            self.last_system_prompt = str(system)
            self.last_user_prompt = str(user)
            return json.dumps({"response": "done", "action": "DONE", "action_arguments": {}})

    provider = FakeProvider()
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)

    store = _sqlite_store(tmp_path)
    try:
        memory = Memory(
            type="project_fact",
            key="memory_anchor",
            value={"note": "remember this preference"},
            scope="project",
            tags=["memory", "integration"],
            source_run_id="run-harness",
        )
        candidate_id = store.put_candidate(memory, provenance="test", rationale="seed")
        store.promote(candidate_id)
        builder = MemoryContextBuilder(store)

        config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
        registry = ToolRegistry()
        registry.register(EchoTool())
        harness = AgentHarness(
            config.agents,
            registry,
            config.llm,
            ToolSandbox(config.agents.sandbox),
            agent_name="research",
            memory_context_builder=builder,
        )

        state = harness.run(
            AgentContext(messages=[{"role": "user", "content": "memory integration topic"}])
        )
        assert state.status.value == "completed"
        assert "Standing Context (Memory)" in provider.last_user_prompt
        assert "memory_anchor" in provider.last_user_prompt

        marker = "Available tools (function calling schema):"
        _, _, payload = provider.last_system_prompt.partition(marker)
        parsed_context = json.loads(payload.strip() or "[]")
        assert isinstance(parsed_context, list)
        assert parsed_context[0]["name"] == "echo"
    finally:
        store.close()
