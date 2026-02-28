from __future__ import annotations

import json
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.models import AgentContext
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
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "require_permission_by_risk": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def _extract_tools_payload(system_prompt: str) -> list[dict]:
    marker = "Available tools (function calling schema):"
    if marker not in system_prompt:
        return []
    _, _, payload = system_prompt.partition(marker)
    return json.loads(payload.strip() or "[]")


def test_harness_uses_function_calling_tools(monkeypatch, tmp_path: Path) -> None:
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
                            {"name": "echo", "arguments": {"text": "hi"}}
                        ],
                    }
                ),
                json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}}),
            ]
            self.received_tools: list[list[dict] | None] = []
            self.received_users: list[str] = []

        def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
            _ = instruction, temperature, max_tokens
            self.received_tools.append(kwargs.get("tools"))
            self.received_users.append(str(input))
            return self.responses.pop(0)

    provider = FunctionProvider()
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="function-calling",
    )

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "say hi"}]))
    assert state.status.value == "completed"
    assert provider.received_tools
    assert provider.received_tools[0] is not None
    assert provider.received_tools[0][0]["name"] == "echo"
    assert "Respond with strict JSON" not in provider.received_users[0]
    assert any(entry.actor == "tool" and entry.action == "echo" for entry in state.log_entries)


def test_harness_falls_back_to_prompt_tools_for_unsupported_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class PromptProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.responses = [
                json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}})
            ]
            self.system_prompts: list[str] = []
            self.user_prompts: list[str] = []
            self.kwargs: list[dict] = []

        def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
            _ = temperature, max_tokens
            self.system_prompts.append(str(instruction))
            self.user_prompts.append(str(input))
            self.kwargs.append(dict(kwargs))
            return self.responses.pop(0)

    provider = PromptProvider()
    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda cfg: provider)

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(EchoTool())
    harness = AgentHarness(
        config.agents,
        registry,
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="prompt-tools",
    )

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "done"}]))
    assert state.status.value == "completed"
    assert provider.system_prompts
    assert provider.user_prompts
    assert all("tools" not in kwargs for kwargs in provider.kwargs)

    tools_payload = _extract_tools_payload(provider.system_prompts[0])
    assert [item["name"] for item in tools_payload] == ["echo"]
    assert "Respond with strict JSON" in provider.user_prompts[0]


def test_parse_tool_response_accepts_tool_calls_string_args(tmp_path: Path) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    harness = AgentHarness(
        config.agents,
        ToolRegistry(),
        config.llm,
        ToolSandbox(config.agents.sandbox),
        agent_name="parse-tools",
    )
    raw = json.dumps(
        {
            "response": "Call tool",
            "tool_calls": [
                {"name": "echo", "arguments": "{\"text\": \"hello\"}"}
            ],
        }
    )
    parsed = harness._parse_tool_response(raw)
    assert parsed.action == "echo"
    assert parsed.action_arguments == {"text": "hello"}
