from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lsm.agents.models import AgentContext
from lsm.agents.productivity.general import GeneralAgent
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


class GeneralAgentWithEcho(GeneralAgent):
    tool_allowlist = set(GeneralAgent.tool_allowlist or set()) | {"echo"}


def _base_raw(tmp_path: Path, *, provider: str, model: str, api_key: str) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "path": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": provider, "api_key": api_key}],
            "services": {"default": {"provider": provider, "model": model}},
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
            "max_tokens_budget": 2000,
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


def test_agent_end_to_end_with_stubbed_llm(monkeypatch, tmp_path: Path) -> None:
    class FunctionProvider:
        name = "fake"
        model = "fake-model"
        supports_function_calling = True

        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "response": "Using tool",
                        "tool_calls": [
                            {"name": "echo", "arguments": {"text": "stubbed"}}
                        ],
                    }
                ),
                json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}}),
            ]

        def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
            system = instruction
            user = input
            _ = instruction, input, temperature, max_tokens, kwargs
            return self.responses.pop(0)

    monkeypatch.setattr(
        "lsm.agents.harness.create_provider",
        lambda cfg: FunctionProvider(),
    )

    config = build_config_from_raw(
        _base_raw(tmp_path, provider="openai", model="gpt-5.2", api_key="test-key"),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgentWithEcho(config.llm, registry, sandbox, config.agents)

    state = agent.run(AgentContext(messages=[{"role": "user", "content": "Echo"}]))
    assert state.status.value == "completed"
    assert any(entry.actor == "tool" and entry.action == "echo" for entry in state.log_entries)


@pytest.mark.live
@pytest.mark.live_llm
def test_agent_end_to_end_with_live_llm(tmp_path: Path) -> None:
    if not os.getenv("LSM_RUN_LIVE_LLM"):
        pytest.skip("Set LSM_RUN_LIVE_LLM=1 to enable live LLM integration test.")

    provider = os.getenv("LSM_LIVE_LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LSM_LIVE_LLM_MODEL", "").strip()
    if not model:
        pytest.skip("Set LSM_LIVE_LLM_MODEL to run live LLM integration test.")

    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)
    if not api_key_env:
        pytest.skip(f"Unsupported provider for live test: {provider}")
    api_key = os.getenv(api_key_env)
    if not api_key:
        pytest.skip(f"Missing {api_key_env} for live LLM integration test.")

    config = build_config_from_raw(
        _base_raw(tmp_path, provider=provider, model=model, api_key=api_key),
        tmp_path / "config.json",
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    agent = GeneralAgentWithEcho(config.llm, registry, sandbox, config.agents)

    prompt = (
        "Call the echo tool with text 'live-ping'. "
        "After the tool returns, respond DONE."
    )
    state = agent.run(AgentContext(messages=[{"role": "user", "content": prompt}]))
    assert state.status.value == "completed"
    assert any(entry.actor == "tool" and entry.action == "echo" for entry in state.log_entries)
