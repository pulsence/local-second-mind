from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.log_formatter import format_agent_log, load_agent_log, save_agent_log
from lsm.agents.models import AgentContext, AgentLogEntry
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo input text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: dict) -> str:
        return str(args.get("text", ""))


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
            "max_tokens_budget": 5000,
            "max_iterations": 10,
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


def test_harness_executes_tool_then_done(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Using tool now",
                        "action": "echo",
                        "action_arguments": {"text": "hello"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Done.",
                        "action": "DONE",
                        "action_arguments": {},
                    }
                ),
            ]

        def synthesize(self, question, context, mode="insight", **kwargs):
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    harness = AgentHarness(config.agents, registry, config.llm, sandbox, agent_name="research")

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "Say hi"}]))
    assert state.status.value == "completed"
    assert len(state.log_entries) >= 3
    assert any(entry.actor == "tool" and entry.action == "echo" for entry in state.log_entries)
    state_file = harness.get_state_path()
    assert state_file is not None
    assert state_file.exists()


def test_harness_log_callback_receives_entries(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Using tool now",
                        "action": "echo",
                        "action_arguments": {"text": "callback"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Done.",
                        "action": "DONE",
                        "action_arguments": {},
                    }
                ),
            ]

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    callback_entries: list[AgentLogEntry] = []

    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        sandbox,
        agent_name="callback",
        log_callback=lambda entry: callback_entries.append(entry),
    )
    state = harness.run(AgentContext(messages=[{"role": "user", "content": "callback"}]))

    assert state.status.value == "completed"
    assert len(callback_entries) == len(state.log_entries)
    assert any(entry.actor == "tool" and entry.action == "echo" for entry in callback_entries)


def test_harness_writes_run_summary_json(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self):
            self._responses = [
                json.dumps(
                    {
                        "response": "Call echo",
                        "action": "echo",
                        "action_arguments": {"text": "hello"},
                    }
                ),
                json.dumps(
                    {
                        "response": "Done",
                        "action": "DONE",
                        "action_arguments": {},
                    }
                ),
            ]

        def synthesize(self, question, context, mode="insight", **kwargs):
            _ = question, context, mode, kwargs
            return self._responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    harness = AgentHarness(config.agents, registry, config.llm, sandbox, agent_name="research")

    state = harness.run(
        AgentContext(
            messages=[
                {
                    "role": "user",
                    "content": "Must cite sources and avoid speculation.",
                }
            ]
        )
    )
    assert state.status.value == "completed"

    state_path = harness.get_state_path()
    assert state_path is not None
    summary_path = state_path.parent / "run_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agent_name"] == "research"
    assert summary["topic"] == "Must cite sources and avoid speculation."
    assert summary["status"] == "completed"
    assert summary["run_outcome"] == "completed"
    assert summary["tools_used"] == {"echo": 1}
    assert summary["tool_sequence"] == ["echo"]
    assert summary["approvals_denials"]["approvals"] == 1
    assert summary["approvals_denials"]["denials"] == 0
    assert summary["approvals_denials"]["by_tool"]["echo"]["approvals"] == 1
    assert summary["token_usage"]["iterations"] == 2
    assert summary["token_usage"]["tokens_used"] > 0
    assert "Must cite sources and avoid speculation" in summary["constraints"]
    assert "avoid speculation" in summary["constraints"]

    assert str(summary_path) in state.artifacts


def test_harness_fresh_context_strategy(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    raw = _base_raw(tmp_path)
    raw["agents"]["context_window_strategy"] = "fresh"
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    harness = AgentHarness(config.agents, registry, config.llm, sandbox, agent_name="fresh")

    initial_messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    prepared = harness._prepare_messages(AgentContext(messages=initial_messages))
    assert len(prepared) == 6

    state = harness.run(AgentContext(messages=initial_messages))
    assert state.status.value == "completed"


def test_harness_background_pause_resume_and_state_save(monkeypatch, tmp_path: Path) -> None:
    started = threading.Event()
    proceed = threading.Event()

    class BlockingProvider:
        name = "fake"
        model = "fake-model"

        def synthesize(self, question, context, mode="insight", **kwargs):
            started.set()
            proceed.wait(timeout=1.5)
            return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: BlockingProvider())

    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    harness = AgentHarness(config.agents, registry, config.llm, sandbox, agent_name="bg")

    thread = harness.start_background(AgentContext(messages=[{"role": "user", "content": "go"}]))
    assert thread.is_alive()
    started.wait(timeout=1.0)
    harness.pause()
    assert harness.state.status.value == "paused"

    state_path = harness.get_state_path()
    assert state_path is not None
    assert state_path.exists()

    harness.resume()
    proceed.set()
    thread.join(timeout=2.0)
    assert harness.state.status.value == "completed"


def test_log_formatter_save_load_and_format(tmp_path: Path) -> None:
    entries = [
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            actor="agent",
            provider_name="openai",
            model_name="gpt-5.2",
            content="Planning step",
            action="query_remote",
            action_arguments={"provider": "arxiv"},
        )
    ]
    output = format_agent_log(entries)
    assert "action=query_remote" in output
    path = save_agent_log(entries, tmp_path / "log.log")
    loaded = load_agent_log(path)
    assert len(loaded) == 1
    assert loaded[0].action == "query_remote"
