from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lsm.agents.harness import AgentHarness
from lsm.agents.log_formatter import format_agent_log, save_agent_log
from lsm.agents.log_redactor import redact_secrets
from lsm.agents.models import AgentContext, AgentLogEntry
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.env_scrubber import scrub_environment
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class SecretTool(BaseTool):
    name = "secret_tool"
    description = "Returns sensitive content."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> str:
        _ = args
        return "token=sk-1234567890abcdef key_value=key_supersecret123"


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
            "max_iterations": 5,
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


def test_scrub_environment_removes_secret_variables() -> None:
    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "TEMP": "/tmp",
        "LANG": "en_US.UTF-8",
        "OPENAI_API_KEY": "sk-abc",
        "MY_SECRET_VALUE": "secret",
        "SESSION_TOKEN": "token",
        "DB_PASSWORD": "password",
        "EXTRA": "drop-me",
    }
    scrubbed = scrub_environment(source)
    assert scrubbed["PATH"] == "/usr/bin"
    assert scrubbed["HOME"] == "/home/user"
    assert scrubbed["TEMP"] == "/tmp"
    assert "OPENAI_API_KEY" not in scrubbed
    assert "MY_SECRET_VALUE" not in scrubbed
    assert "SESSION_TOKEN" not in scrubbed
    assert "DB_PASSWORD" not in scrubbed
    assert "EXTRA" not in scrubbed


def test_scrub_environment_preserves_minimal_runtime_keys() -> None:
    source = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "USERPROFILE": "C:\\Users\\user",
        "TEMP": "/tmp",
        "TMP": "/tmp2",
        "LANG": "en_US.UTF-8",
    }
    scrubbed = scrub_environment(source)
    assert scrubbed["PATH"] == "/usr/bin"
    assert scrubbed["HOME"] == "/home/user"
    assert scrubbed["USERPROFILE"] == "C:\\Users\\user"
    assert scrubbed["TEMP"] == "/tmp"
    assert scrubbed["TMP"] == "/tmp2"
    assert scrubbed["LANG"] == "en_US.UTF-8"


def test_scrub_environment_drops_non_minimal_variables() -> None:
    source = {"PATH": "/usr/bin", "HOME": "/home/user", "PYTHONPATH": "/custom/python"}
    scrubbed = scrub_environment(source)
    assert "PYTHONPATH" not in scrubbed


def test_log_redactor_masks_secret_patterns() -> None:
    text = "OPENAI_API_KEY=sk-1234567890abcdef and key_verysecret123 plus AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    redacted = redact_secrets(text)
    assert "sk-1234567890abcdef" not in redacted
    assert "key_verysecret123" not in redacted
    assert "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in redacted
    assert "[REDACTED]" in redacted


def test_log_redactor_masks_base64_blobs() -> None:
    blob = "QWxhZGRpbjpvcGVuIHNlc2FtZSBzdXBlciBzZWNyZXQhQWxhZGRpbjpvcGVuIHNlc2FtZQ=="
    redacted = redact_secrets(f"value={blob}")
    assert blob not in redacted
    assert "[REDACTED]" in redacted


def test_save_agent_log_redacts_content(tmp_path: Path) -> None:
    entries = [
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            actor="agent",
            content="API_KEY=sk-1234567890abcdef",
        )
    ]
    path = save_agent_log(entries, tmp_path / "agent_log.log")
    text = path.read_text(encoding="utf-8")
    assert "API_KEY=[REDACTED]" in text
    assert "sk-1234567890abcdef" not in text


def test_format_agent_log_redacts_content() -> None:
    entries = [
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            actor="agent",
            content="OPENAI_API_KEY=sk-1234567890abcdef",
        )
    ]
    output = format_agent_log(entries)
    assert "sk-1234567890abcdef" not in output
    assert "[REDACTED]" in output


def test_saved_logs_contain_no_plaintext_secret_values(tmp_path: Path) -> None:
    entries = [
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            actor="agent",
            content="token=sk-1234567890abcdef",
        ),
        AgentLogEntry(
            timestamp=datetime.utcnow(),
            actor="tool",
            content="secret=key_topsecret987",
        ),
    ]
    path = save_agent_log(entries, tmp_path / "security_log.log")
    raw = path.read_text(encoding="utf-8")
    assert "sk-1234567890abcdef" not in raw
    assert "key_topsecret987" not in raw


def test_harness_redacts_tool_output_before_logging(monkeypatch, tmp_path: Path) -> None:
    class FakeProvider:
        name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.responses = [
                json.dumps(
                    {
                        "response": "Calling tool",
                        "action": "secret_tool",
                        "action_arguments": {},
                    }
                ),
                json.dumps({"response": "Done", "action": "DONE", "action_arguments": {}}),
            ]

        def send_message(self, input, instruction=None, prompt=None, temperature=None, max_tokens=4096, previous_response_id=None, prompt_cache_key=None, prompt_cache_retention=None, **kwargs):
            _ = instruction, input, temperature, max_tokens, kwargs
            return self.responses.pop(0)

    monkeypatch.setattr("lsm.agents.harness.create_provider", lambda config: FakeProvider())

    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    registry = ToolRegistry()
    registry.register(SecretTool())
    sandbox = ToolSandbox(config.agents.sandbox)
    harness = AgentHarness(config.agents, registry, config.llm, sandbox, agent_name="secret")

    state = harness.run(AgentContext(messages=[{"role": "user", "content": "run"}]))
    tool_entries = [entry for entry in state.log_entries if entry.actor == "tool"]
    assert len(tool_entries) == 1
    assert "sk-1234567890abcdef" not in tool_entries[0].content
    assert "key_supersecret123" not in tool_entries[0].content
    assert "[REDACTED]" in tool_entries[0].content

    state_path = harness.get_state_path()
    assert state_path is not None
    persisted = state_path.read_text(encoding="utf-8")
    assert "sk-1234567890abcdef" not in persisted
    assert "key_supersecret123" not in persisted
