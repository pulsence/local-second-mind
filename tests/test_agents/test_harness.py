from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from lsm.agents.harness import AgentHarness
from lsm.agents.log_formatter import format_agent_log, load_agent_log, save_agent_log
from lsm.agents.models import AgentContext, AgentLogEntry
from lsm.agents.phase import PhaseResult
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

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
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

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
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

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
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

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
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
    prepared = harness._prepare_messages_from_history(initial_messages)
    assert len(prepared) == 6

    state = harness.run(AgentContext(messages=initial_messages))
    assert state.status.value == "completed"


def test_harness_background_pause_resume_and_state_save(monkeypatch, tmp_path: Path) -> None:
    started = threading.Event()
    proceed = threading.Event()

    class BlockingProvider:
        name = "fake"
        model = "fake-model"

        def _send_message(self, system, user, temperature, max_tokens, **kwargs):
            _ = system, user, temperature, max_tokens, kwargs
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
            action="query_arxiv",
            action_arguments={"provider": "arxiv"},
        )
    ]
    output = format_agent_log(entries)
    assert "action=query_arxiv" in output
    path = save_agent_log(entries, tmp_path / "log.log")
    loaded = load_agent_log(path)
    assert len(loaded) == 1
    assert loaded[0].action == "query_arxiv"


# ---------------------------------------------------------------------------
# run_bounded() tests
# ---------------------------------------------------------------------------


class _RecordingProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.call_count = 0

    def _send_message(self, system, user, temperature, max_tokens, **kwargs):
        _ = system, user, temperature, max_tokens, kwargs
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})


def _done_response() -> str:
    return json.dumps({"response": "Done.", "action": "DONE", "action_arguments": {}})


def _echo_response() -> str:
    return json.dumps({"response": "Calling echo", "action": "echo", "action_arguments": {"text": "hello"}})


def _make_bounded_harness(
    tmp_path: Path,
    *,
    tool_allowlist: set[str] | None = None,
) -> AgentHarness:
    raw = _base_raw(tmp_path)
    config = build_config_from_raw(raw, tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    return AgentHarness(
        config.agents,
        registry,
        config.llm,
        sandbox,
        agent_name="test",
        tool_allowlist=tool_allowlist,
    )


def test_run_bounded_returns_done_when_llm_signals_done(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi")

    assert result.stop_reason == "done"
    assert result.final_text == "Done."
    assert result.tool_calls == []


def test_run_bounded_returns_max_iterations_at_limit(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_echo_response(), _echo_response(), _echo_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi", max_iterations=2)

    assert result.stop_reason == "max_iterations"
    assert provider.call_count == 2


def test_run_bounded_returns_budget_exhausted_at_entry(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)
    harness.context = AgentContext(
        messages=[],
        budget_tracking={"max_tokens_budget": 100, "tokens_used": 200},
    )

    result = harness.run_bounded(user_message="hi")

    assert result.stop_reason == "budget_exhausted"
    assert provider.call_count == 0


def test_run_bounded_subsequent_call_also_budget_exhausted(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)
    harness.context = AgentContext(
        messages=[],
        budget_tracking={"max_tokens_budget": 100, "tokens_used": 200},
    )

    r1 = harness.run_bounded(user_message="first")
    r2 = harness.run_bounded(user_message="second")

    assert r1.stop_reason == "budget_exhausted"
    assert r2.stop_reason == "budget_exhausted"
    assert provider.call_count == 0


def test_run_bounded_returns_stop_requested(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)
    harness._stop_event.set()

    result = harness.run_bounded(user_message="hi")

    assert result.stop_reason == "stop_requested"
    assert provider.call_count == 0


def test_run_bounded_phase_result_has_no_financial_fields(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi")

    assert isinstance(result, PhaseResult)
    with pytest.raises(AttributeError):
        _ = result.tokens_used  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = result.cost_usd  # type: ignore[attr-defined]


def test_run_bounded_continue_context_true_includes_prior_history(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    harness.run_bounded(user_message="query_A", context_label="ctx")
    harness.run_bounded(user_message="query_B", context_label="ctx", continue_context=True)

    history = harness._context_histories["ctx"]
    contents = [str(m.get("content", "")) for m in history]
    assert any("query_A" in c for c in contents)
    assert any("query_B" in c for c in contents)


def test_run_bounded_continue_context_false_resets_history(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    harness.run_bounded(user_message="query_A", context_label="ctx")
    harness.run_bounded(user_message="query_B", context_label="ctx", continue_context=False)

    history = harness._context_histories["ctx"]
    contents = [str(m.get("content", "")) for m in history]
    assert not any("query_A" in c for c in contents)
    assert any("query_B" in c for c in contents)


def test_run_bounded_independent_context_labels(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    harness.run_bounded(user_message="msg_a", context_label="label_a")
    harness.run_bounded(user_message="msg_b", context_label="label_b")

    a_contents = [str(m.get("content", "")) for m in harness._context_histories["label_a"]]
    b_contents = [str(m.get("content", "")) for m in harness._context_histories["label_b"]]
    assert any("msg_a" in c for c in a_contents)
    assert not any("msg_a" in c for c in b_contents)
    assert any("msg_b" in c for c in b_contents)
    assert not any("msg_b" in c for c in a_contents)


def test_run_bounded_context_label_none_separate_from_named(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    harness.run_bounded(user_message="primary", context_label=None)
    harness.run_bounded(user_message="named_msg", context_label="named")

    none_contents = [str(m.get("content", "")) for m in harness._context_histories.get(None, [])]
    named_contents = [str(m.get("content", "")) for m in harness._context_histories.get("named", [])]
    assert any("primary" in c for c in none_contents)
    assert not any("named_msg" in c for c in none_contents)
    assert any("named_msg" in c for c in named_contents)
    assert not any("primary" in c for c in named_contents)


def test_run_bounded_tool_names_restricts_to_subset(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_echo_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi", tool_names=["nonexistent"])

    assert result.stop_reason == "done"
    assert len(result.tool_calls) == 1
    assert "error" in result.tool_calls[0]
    assert "not in the requested tool_names list" in result.tool_calls[0]["error"]


def test_run_bounded_tool_names_empty_rejects_all_tool_calls(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_echo_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi", tool_names=[])

    assert result.stop_reason == "done"
    assert any("error" in tc for tc in result.tool_calls)


def test_run_bounded_tool_names_none_all_tools_available(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_echo_response(), _done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(user_message="hi", tool_names=None)

    assert result.stop_reason == "done"
    assert len(result.tool_calls) == 1
    assert "result" in result.tool_calls[0]
    assert result.tool_calls[0]["result"] == "hello"


def test_run_bounded_direct_tool_calls_executes_without_llm(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([_done_response()])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(direct_tool_calls=[{"name": "echo", "arguments": {"text": "hi"}}])

    assert result.stop_reason == "done"
    assert result.final_text == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["result"] == "hi"
    assert provider.call_count == 0


def test_run_bounded_direct_tool_calls_empty_returns_done(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)

    result = harness.run_bounded(direct_tool_calls=[])

    assert result.stop_reason == "done"
    assert result.tool_calls == []
    assert result.final_text == ""
    assert provider.call_count == 0


def test_run_bounded_direct_tool_calls_stop_requested_returns_immediately(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)
    harness._stop_event.set()

    result = harness.run_bounded(direct_tool_calls=[{"name": "echo", "arguments": {"text": "hi"}}])

    assert result.stop_reason == "stop_requested"
    assert result.tool_calls == []
    assert provider.call_count == 0


def test_run_bounded_direct_tool_calls_budget_irrelevant(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path)
    harness.context = AgentContext(
        messages=[],
        budget_tracking={"max_tokens_budget": 1, "tokens_used": 1000},
    )

    result = harness.run_bounded(direct_tool_calls=[{"name": "echo", "arguments": {"text": "hi"}}])

    assert result.stop_reason == "done"
    assert result.tool_calls[0]["result"] == "hi"


def test_run_bounded_direct_tool_calls_disallowed_tool_produces_error(monkeypatch, tmp_path: Path) -> None:
    provider = _RecordingProvider([])
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)
    harness = _make_bounded_harness(tmp_path, tool_allowlist={"some_other_tool"})

    result = harness.run_bounded(direct_tool_calls=[{"name": "echo", "arguments": {"text": "hi"}}])

    assert result.stop_reason == "done"
    assert len(result.tool_calls) == 1
    assert "error" in result.tool_calls[0]
    assert "not allowed" in result.tool_calls[0]["error"]
