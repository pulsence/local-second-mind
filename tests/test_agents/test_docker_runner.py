from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.docker_runner import DockerRunner

pytestmark = pytest.mark.docker


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo tool."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        return str(args["text"])


def test_docker_runner_builds_restricted_command_and_parses_output(monkeypatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = capture_output, text, check
        captured["command"] = list(command)
        captured["input"] = input
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "stdout": "hello from docker",
                    "stderr": "",
                    "artifacts": ["/workspace/output.txt"],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("lsm.agents.tools.docker_runner.shutil.which", lambda _: "docker")
    monkeypatch.setattr("lsm.agents.tools.docker_runner.subprocess.run", fake_run)

    read_root = tmp_path / "docs"
    read_root.mkdir()
    runner = DockerRunner(
        image="lsm-agent-sandbox:test",
        workspace_root=tmp_path,
        read_paths=[read_root],
        timeout_s_default=12,
        max_stdout_kb=64,
        network_default="none",
        cpu_limit=2.0,
        mem_limit_mb=256,
        read_only_root=True,
        pids_limit=128,
    )
    result = runner.run(EchoTool(), {"text": "hello"}, {})

    assert result.stdout == "hello from docker"
    assert result.stderr == ""
    assert result.runner_used == "docker"
    assert result.runtime_ms >= 0
    assert result.artifacts == ["/workspace/output.txt"]

    command = captured["command"]
    assert command[0] == "docker"
    assert "run" in command
    assert "--read-only" in command
    assert "--network" in command
    assert "none" in command
    assert "--cpus" in command
    assert "2.0" in command
    assert "--memory" in command
    assert "256m" in command
    assert "--pids-limit" in command
    assert "128" in command
    assert "lsm-agent-sandbox:test" in command
    assert "python" in command
    assert "-m" in command
    assert "lsm.agents.tools._docker_entrypoint" in command
    assert any("dst=/workspace,rw" in token for token in command)
    assert any("dst=/sandbox/ro/0,readonly" in token for token in command)

    payload = json.loads(captured["input"])
    assert payload["tool_name"] == "echo"
    assert payload["tool_module"] == EchoTool.__module__
    assert payload["tool_class"] == EchoTool.__name__
    assert payload["args"] == {"text": "hello"}


def test_docker_runner_raises_when_docker_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("lsm.agents.tools.docker_runner.shutil.which", lambda _: None)
    runner = DockerRunner(workspace_root=tmp_path)
    with pytest.raises(RuntimeError, match="Docker CLI is not available"):
        runner.run(EchoTool(), {"text": "hello"}, {})


def test_docker_runner_enforces_timeout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = command, input, capture_output, text, timeout, check
        raise subprocess.TimeoutExpired(cmd="docker run", timeout=0.1)

    monkeypatch.setattr("lsm.agents.tools.docker_runner.shutil.which", lambda _: "docker")
    monkeypatch.setattr("lsm.agents.tools.docker_runner.subprocess.run", fake_run)
    runner = DockerRunner(workspace_root=tmp_path, timeout_s_default=0.1)
    with pytest.raises(TimeoutError, match="exceeded timeout"):
        runner.run(EchoTool(), {"text": "hello"}, {})


def test_docker_runner_raises_on_nonzero_exit(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = input, capture_output, text, timeout, check
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="container failed",
        )

    monkeypatch.setattr("lsm.agents.tools.docker_runner.shutil.which", lambda _: "docker")
    monkeypatch.setattr("lsm.agents.tools.docker_runner.subprocess.run", fake_run)
    runner = DockerRunner(workspace_root=tmp_path)
    with pytest.raises(RuntimeError, match="container failed"):
        runner.run(EchoTool(), {"text": "hello"}, {})
