"""
Tests for AgentHarness context and conversation chain tracking.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
from lsm.agents.phase import PhaseResult
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            "max_tokens_budget": 15000,
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


def _build_harness(tmp_path: Path) -> AgentHarness:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    sandbox = ToolSandbox(config.agents.sandbox)
    return AgentHarness(
        agent_config=config.agents,
        tool_registry=registry,
        llm_registry=config.llm,
        sandbox=sandbox,
        agent_name="test_agent",
        lsm_config=config,
    )


# ---------------------------------------------------------------------------
# Tests: context_chain_state initialization
# ---------------------------------------------------------------------------


def test_chain_state_initialized_empty(tmp_path: Path) -> None:
    harness = _build_harness(tmp_path)
    assert harness._context_chain_state == {}


def test_chain_state_created_on_run_bounded(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    # Mock _call_llm to return a simple "DONE" response
    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        harness._last_llm_response_id = "resp-001"
        return "I'm done. DONE.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
            "send_message": lambda *a, **kw: "DONE"
        })(),
    )

    result = harness.run_bounded(
        user_message="hello",
        max_iterations=1,
        continue_context=False,
        context_label="label_a",
    )

    assert "label_a" in harness._context_chain_state
    assert harness._context_chain_state["label_a"].get("last_response_id") == "resp-001"


def test_chain_state_persists_across_calls_same_label(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    call_count = [0]
    captured_prev_ids = []

    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        captured_prev_ids.append(previous_response_id)
        call_count[0] += 1
        harness._last_llm_response_id = f"resp-{call_count[0]:03d}"
        return "Done.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
        })(),
    )

    # First call with label_a
    harness.run_bounded(
        user_message="first",
        max_iterations=1,
        continue_context=True,
        context_label="label_a",
    )
    # Second call with same label — should carry forward response_id
    harness.run_bounded(
        user_message="second",
        max_iterations=1,
        continue_context=True,
        context_label="label_a",
    )

    assert captured_prev_ids[0] is None  # First call has no prior
    assert captured_prev_ids[1] == "resp-001"  # Second call carries forward


def test_chain_state_reset_on_continue_context_false(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    call_count = [0]
    captured_prev_ids = []

    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        captured_prev_ids.append(previous_response_id)
        call_count[0] += 1
        harness._last_llm_response_id = f"resp-{call_count[0]:03d}"
        return "Done.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
        })(),
    )

    # First call
    harness.run_bounded(
        user_message="first",
        max_iterations=1,
        continue_context=True,
        context_label="label_a",
    )
    # Reset context — should clear chain state
    harness.run_bounded(
        user_message="fresh start",
        max_iterations=1,
        continue_context=False,
        context_label="label_a",
    )

    assert captured_prev_ids[0] is None
    assert captured_prev_ids[1] is None  # Reset cleared the chain


def test_separate_labels_isolated(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    call_count = [0]
    captured = []

    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        captured.append({
            "prev_id": previous_response_id,
            "cache_key": prompt_cache_key,
        })
        call_count[0] += 1
        harness._last_llm_response_id = f"resp-{call_count[0]:03d}"
        return "Done.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
        })(),
    )

    # Label A gets resp-001
    harness.run_bounded(
        user_message="a",
        max_iterations=1,
        continue_context=True,
        context_label="label_a",
    )
    # Label B — should NOT see label_a's response_id
    harness.run_bounded(
        user_message="b",
        max_iterations=1,
        continue_context=True,
        context_label="label_b",
    )
    # Label A again — should carry forward resp-001
    harness.run_bounded(
        user_message="a again",
        max_iterations=1,
        continue_context=True,
        context_label="label_a",
    )

    assert captured[0]["prev_id"] is None  # label_a first call
    assert captured[1]["prev_id"] is None  # label_b first call — isolated
    assert captured[2]["prev_id"] == "resp-001"  # label_a carries forward


def test_none_label_isolated_from_named(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    call_count = [0]
    captured_prev_ids = []

    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        captured_prev_ids.append(previous_response_id)
        call_count[0] += 1
        harness._last_llm_response_id = f"resp-{call_count[0]:03d}"
        return "Done.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
        })(),
    )

    # Named label gets resp-001
    harness.run_bounded(
        user_message="named",
        max_iterations=1,
        continue_context=True,
        context_label="named",
    )
    # None label — should not see named's response_id
    harness.run_bounded(
        user_message="unnamed",
        max_iterations=1,
        continue_context=True,
        context_label=None,
    )

    assert captured_prev_ids[0] is None
    assert captured_prev_ids[1] is None  # None label is isolated


def test_prompt_cache_key_includes_label(tmp_path: Path, monkeypatch) -> None:
    harness = _build_harness(tmp_path)
    harness._ensure_context()

    captured_keys = []

    def fake_call_llm(llm_provider, *, messages_for_llm, standing_context_block,
                      tool_definitions=None, previous_response_id=None,
                      prompt_cache_key=None):
        captured_keys.append(prompt_cache_key)
        harness._last_llm_response_id = None
        return "Done.", "system", "user"

    monkeypatch.setattr(harness, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: type("P", (), {
            "name": "test", "model": "test",
        })(),
    )

    harness.run_bounded(
        user_message="test",
        max_iterations=1,
        continue_context=True,
        context_label="my_label",
    )

    assert captured_keys[0] is not None
    assert "my_label" in captured_keys[0]
    assert "test_agent" in captured_keys[0]
