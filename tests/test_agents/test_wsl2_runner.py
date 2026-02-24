from __future__ import annotations

import json
import subprocess
from typing import Any, Dict

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.wsl2_runner import WSL2Runner


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


def test_wsl2_runner_translates_paths_and_scrubs_env(monkeypatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = capture_output, text, timeout, check
        captured["command"] = list(command)
        captured["input"] = input
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps({"stdout": "ok", "stderr": "", "artifacts": []}),
            stderr="",
        )

    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.shutil.which", lambda _: "wsl")
    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.subprocess.run", fake_run)

    runner = WSL2Runner(
        workspace_root="C:\\Users\\Test\\workspace",
        distro="Ubuntu",
    )
    runner.run(
        EchoTool(),
        {"path": "C:\\Users\\Test\\docs\\note.txt"},
        {"PATH": "C:\\Windows\\System32", "OPENAI_API_KEY": "secret"},
    )

    command = captured["command"]
    assert command[0] == "wsl"
    assert "-d" in command
    assert "Ubuntu" in command
    assert "bash" in command
    assert "-lc" in command
    script = command[-1]
    assert "/mnt/c/Users/Test/workspace" in script
    assert "python -m lsm.agents.tools._docker_entrypoint" in script
    assert "OPENAI_API_KEY" not in script

    payload = json.loads(captured["input"])
    assert payload["args"]["path"] == "/mnt/c/Users/Test/docs/note.txt"


def test_wsl2_runner_raises_when_wsl_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.shutil.which", lambda _: None)
    runner = WSL2Runner()
    with pytest.raises(RuntimeError, match="WSL2 CLI is not available"):
        runner.run(EchoTool(), {"text": "hello"}, {})


def test_wsl2_runner_enforces_timeout(monkeypatch) -> None:
    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = command, input, capture_output, text, timeout, check
        raise subprocess.TimeoutExpired(cmd="wsl", timeout=0.1)

    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.shutil.which", lambda _: "wsl")
    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.subprocess.run", fake_run)
    runner = WSL2Runner(timeout_s_default=0.1)
    with pytest.raises(TimeoutError, match="exceeded timeout"):
        runner.run(EchoTool(), {"text": "hello"}, {})


def test_wsl2_runner_raises_on_nonzero_exit(monkeypatch) -> None:
    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = input, capture_output, text, timeout, check
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="wsl failed",
        )

    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.shutil.which", lambda _: "wsl")
    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.subprocess.run", fake_run)
    runner = WSL2Runner()
    with pytest.raises(RuntimeError, match="wsl failed"):
        runner.run(EchoTool(), {"text": "hello"}, {})


def test_wsl2_runner_raises_on_invocation_error(monkeypatch) -> None:
    def fake_run(command, *, input, capture_output, text, timeout, check):
        _ = command, input, capture_output, text, timeout, check
        raise OSError("boom")

    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.shutil.which", lambda _: "wsl")
    monkeypatch.setattr("lsm.agents.tools.wsl2_runner.subprocess.run", fake_run)
    runner = WSL2Runner()
    with pytest.raises(RuntimeError, match="WSL2 runner failed to start"):
        runner.run(EchoTool(), {"text": "hello"}, {})
