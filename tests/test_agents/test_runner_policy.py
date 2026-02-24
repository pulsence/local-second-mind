from __future__ import annotations

from typing import Any, Dict, Mapping

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.runner import BaseRunner, ToolExecutionResult
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


class ReadTool(BaseTool):
    name = "read_file"
    description = "Read tool."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        return f"read:{args['path']}"


class WriteTool(BaseTool):
    name = "write_file"
    description = "Write tool."
    risk_level = "writes_workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        return f"write:{args['path']}"


class NetworkTool(BaseTool):
    name = "query_remote"
    description = "Network tool."
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: Dict[str, Any]) -> str:
        _ = args
        return "network"


class ExecTool(BaseTool):
    name = "exec_tool"
    description = "Exec tool."
    risk_level = "exec"
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: Dict[str, Any]) -> str:
        _ = args
        return "exec"


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


def test_runner_policy_read_only_uses_local_without_force_docker(tmp_path) -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            execution_mode="prefer_docker",
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(ReadTool(), {"path": str(tmp_path / "file.txt")})
    assert output == "local"
    assert local.calls == 1
    assert docker.calls == 0


def test_runner_policy_writes_workspace_uses_local_without_force_docker(tmp_path) -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_write_paths=[tmp_path],
            execution_mode="prefer_docker",
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(
        WriteTool(),
        {"path": str(tmp_path / "out.txt"), "content": "hello"},
    )
    assert output == "local"
    assert local.calls == 1
    assert docker.calls == 0


def test_runner_policy_force_docker_routes_read_only_to_docker(tmp_path) -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            execution_mode="prefer_docker",
            force_docker=True,
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(ReadTool(), {"path": str(tmp_path / "file.txt")})
    assert output == "docker"
    assert local.calls == 0
    assert docker.calls == 1


def test_runner_policy_force_docker_routes_write_to_docker(tmp_path) -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_write_paths=[tmp_path],
            execution_mode="prefer_docker",
            force_docker=True,
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(
        WriteTool(),
        {"path": str(tmp_path / "out.txt"), "content": "hello"},
    )
    assert output == "docker"
    assert local.calls == 0
    assert docker.calls == 1


def test_runner_policy_force_docker_blocks_when_docker_unavailable(tmp_path) -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            force_docker=True,
            docker={"enabled": False},
        ),
        local_runner=local,
        docker_runner=None,
    )
    with pytest.raises(PermissionError, match="requires Docker execution"):
        sandbox.execute(ReadTool(), {"path": str(tmp_path / "file.txt")})
    assert local.calls == 0


def test_runner_policy_network_uses_docker_when_available() -> None:
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
    assert local.calls == 0
    assert docker.calls == 1


def test_runner_policy_exec_uses_docker_when_available() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    docker = RecordingRunner(ToolExecutionResult(stdout="docker", runner_used="docker"))
    sandbox = ToolSandbox(
        SandboxConfig(
            execution_mode="prefer_docker",
            docker={"enabled": True},
        ),
        local_runner=local,
        docker_runner=docker,
    )
    output = sandbox.execute(ExecTool(), {})
    assert output == "docker"
    assert local.calls == 0
    assert docker.calls == 1


def test_runner_policy_network_requires_confirmation_without_docker() -> None:
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


def test_runner_policy_exec_requires_confirmation_without_docker() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    sandbox = ToolSandbox(
        SandboxConfig(
            execution_mode="prefer_docker",
            docker={"enabled": False},
        ),
        local_runner=local,
        docker_runner=None,
    )
    with pytest.raises(PermissionError, match="requires user confirmation"):
        sandbox.execute(ExecTool(), {})
    assert local.calls == 0


def test_runner_policy_prefers_wsl2_when_docker_unavailable() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    wsl2 = RecordingRunner(ToolExecutionResult(stdout="wsl2", runner_used="wsl2"))
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            execution_mode="prefer_docker",
            docker={"enabled": False},
            wsl2={"enabled": True},
        ),
        local_runner=local,
        docker_runner=None,
        wsl2_runner=wsl2,
    )
    output = sandbox.execute(NetworkTool(), {})
    assert output == "wsl2"
    assert local.calls == 0
    assert wsl2.calls == 1


def test_runner_policy_local_only_ignores_wsl2() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    wsl2 = RecordingRunner(ToolExecutionResult(stdout="wsl2", runner_used="wsl2"))
    sandbox = ToolSandbox(
        SandboxConfig(
            execution_mode="local_only",
            docker={"enabled": False},
            wsl2={"enabled": True},
        ),
        local_runner=local,
        docker_runner=None,
        wsl2_runner=wsl2,
    )
    output = sandbox.execute(ExecTool(), {})
    assert output == "local"
    assert local.calls == 1
    assert wsl2.calls == 0


def test_runner_policy_force_docker_blocks_even_with_wsl2() -> None:
    local = RecordingRunner(ToolExecutionResult(stdout="local", runner_used="local"))
    wsl2 = RecordingRunner(ToolExecutionResult(stdout="wsl2", runner_used="wsl2"))
    sandbox = ToolSandbox(
        SandboxConfig(
            execution_mode="prefer_docker",
            force_docker=True,
            docker={"enabled": False},
            wsl2={"enabled": True},
        ),
        local_runner=local,
        docker_runner=None,
        wsl2_runner=wsl2,
    )
    with pytest.raises(PermissionError, match="requires Docker execution"):
        sandbox.execute(ExecTool(), {})
    assert local.calls == 0
    assert wsl2.calls == 0


def test_runner_policy_wsl2_unavailable_blocks(monkeypatch) -> None:
    from lsm.agents.tools.wsl2_runner import WSL2Runner

    monkeypatch.setattr(WSL2Runner, "is_available", lambda self: False)
    sandbox = ToolSandbox(
        SandboxConfig(
            execution_mode="prefer_docker",
            docker={"enabled": False},
            wsl2={"enabled": True},
        ),
        docker_runner=None,
    )
    with pytest.raises(PermissionError, match="requires user confirmation"):
        sandbox.execute(ExecTool(), {})
