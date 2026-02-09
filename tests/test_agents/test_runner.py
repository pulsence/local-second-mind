from __future__ import annotations

import time
from typing import Any, Dict, Mapping

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.runner import BaseRunner, LocalRunner, ToolExecutionResult
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        return str(args["text"])


class SlowTool(BaseTool):
    name = "slow"
    description = "Slow tool."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: Dict[str, Any]) -> str:
        _ = args
        time.sleep(0.2)
        return "done"


class WriteTool(BaseTool):
    name = "write_file"
    description = "Write tool."
    risk_level = "writes_workspace"
    requires_permission = False
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        return f"wrote:{args['path']}"


class NetworkTool(BaseTool):
    name = "query_remote"
    description = "Network tool."
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: Dict[str, Any]) -> str:
        _ = args
        return "network"


class RecordingRunner(BaseRunner):
    def __init__(self, result: ToolExecutionResult) -> None:
        self.result = result
        self.last_env: Mapping[str, str] | None = None
        self.calls = 0

    def run(self, tool: BaseTool, args: Dict[str, Any], env: Mapping[str, str]) -> ToolExecutionResult:
        _ = tool, args
        self.calls += 1
        self.last_env = dict(env)
        return ToolExecutionResult(
            stdout=self.result.stdout,
            stderr=self.result.stderr,
            runner_used=self.result.runner_used,
            runtime_ms=self.result.runtime_ms,
            artifacts=list(self.result.artifacts),
        )


def test_local_runner_executes_tool_and_returns_result() -> None:
    runner = LocalRunner(timeout_s_default=1, max_stdout_kb=64, max_file_write_mb=4)
    result = runner.run(EchoTool(), {"text": "hello"}, {})
    assert result.stdout == "hello"
    assert result.stderr == ""
    assert result.runner_used == "local"
    assert result.runtime_ms >= 0
    assert result.artifacts == []


def test_local_runner_truncates_stdout_by_byte_limit() -> None:
    runner = LocalRunner(timeout_s_default=1, max_stdout_kb=1, max_file_write_mb=4)
    result = runner.run(EchoTool(), {"text": "x" * 3000}, {})
    assert "[TRUNCATED " in result.stdout
    assert " bytes]" in result.stdout


def test_local_runner_enforces_timeout() -> None:
    runner = LocalRunner(timeout_s_default=0.05, max_stdout_kb=64, max_file_write_mb=4)
    with pytest.raises(TimeoutError, match="exceeded timeout"):
        runner.run(SlowTool(), {}, {})


def test_local_runner_enforces_max_file_write_size() -> None:
    runner = LocalRunner(timeout_s_default=1, max_stdout_kb=64, max_file_write_mb=0.0005)
    with pytest.raises(ValueError, match="max_file_write_mb"):
        runner.run(
            WriteTool(),
            {"path": "out.txt", "content": "x" * 2000},
            {},
        )


def test_sandbox_executes_via_runner_with_scrubbed_env_and_redaction(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
    recording = RecordingRunner(
        ToolExecutionResult(
            stdout="OPENAI_API_KEY=sk-1234567890abcdef",
            runner_used="local",
            artifacts=["C:\\tmp\\artifact.txt"],
        )
    )
    sandbox = ToolSandbox(
        SandboxConfig(),
        local_runner=recording,
    )
    output = sandbox.execute(EchoTool(), {"text": "hi"})
    assert "sk-1234567890abcdef" not in output
    assert "[REDACTED]" in output
    assert recording.last_env is not None
    assert "OPENAI_API_KEY" not in recording.last_env
    assert sandbox.last_execution_result is not None
    assert sandbox.last_execution_result.artifacts == ["C:\\tmp\\artifact.txt"]


def test_sandbox_selects_docker_runner_for_network_risk_when_enabled() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            execution_mode="prefer_docker",
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(NetworkTool(), {})
    assert output == "docker"
    assert docker.calls == 1
    assert local.calls == 0
    assert sandbox.last_execution_result is not None
    assert sandbox.last_execution_result.runner_used == "docker"


def test_sandbox_requires_confirmation_when_prefer_docker_unavailable_for_network_risk() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            execution_mode="prefer_docker",
            docker={"enabled": False},
        ),
        local_runner=local,
        docker_runner=None,
    )
    with pytest.raises(PermissionError, match="requires user confirmation"):
        sandbox.execute(NetworkTool(), {})
    assert local.calls == 0
    assert sandbox.last_execution_result is None


def test_sandbox_uses_local_for_network_risk_when_execution_mode_is_local_only() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            execution_mode="local_only",
            docker={"enabled": False},
        ),
        local_runner=local,
        docker_runner=None,
    )
    output = sandbox.execute(NetworkTool(), {})
    assert output == "local"
    assert local.calls == 1
    assert sandbox.last_execution_result is not None
    assert sandbox.last_execution_result.runner_used == "local"
