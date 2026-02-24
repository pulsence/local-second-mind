from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lsm.agents.tools.bash import BashTool
from lsm.agents.tools.powershell import PowerShellTool
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


def _patch_shell_runner(monkeypatch) -> None:
    def fake_run(command, *, capture_output, text, timeout, check):
        _ = command, capture_output, text, timeout, check
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("lsm.agents.tools.bash.subprocess.run", fake_run)
    monkeypatch.setattr("lsm.agents.tools.powershell.subprocess.run", fake_run)


def test_security_command_tools_block_path_traversal(tmp_path: Path, monkeypatch) -> None:
    _patch_shell_runner(monkeypatch)
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            command_allowlist=["cat", "get-content"],
        )
    )
    with pytest.raises(PermissionError, match="Path traversal"):
        sandbox.execute(BashTool(), {"command": "cat ../secret.txt"})
    with pytest.raises(PermissionError, match="Path traversal"):
        sandbox.execute(PowerShellTool(), {"command": "Get-Content ..\\secret.txt"})


@pytest.mark.parametrize(
    "command",
    [
        "ls; rm -rf /",
        "ls && whoami",
        "ls || whoami",
        "ls | whoami",
        "echo `whoami`",
        "echo $(whoami)",
        "echo $OPENAI_API_KEY",
    ],
)
def test_security_command_tools_block_injection_vectors(
    tmp_path: Path,
    monkeypatch,
    command: str,
) -> None:
    _patch_shell_runner(monkeypatch)
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            command_allowlist=["ls", "echo"],
        )
    )
    with pytest.raises(PermissionError, match="Command chaining"):
        sandbox.execute(BashTool(), {"command": command})


def test_security_command_tools_allowlist_and_denylist_enforced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_shell_runner(monkeypatch)
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            command_allowlist=["ls"],
            command_denylist=["rm"],
        )
    )
    assert sandbox.execute(BashTool(), {"command": "/bin/LS -la"}) == "ok"
    with pytest.raises(PermissionError, match="denylist"):
        sandbox.execute(BashTool(), {"command": "/bin/RM -rf /"})


def test_security_command_tools_block_write_redirection_outside_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_shell_runner(monkeypatch)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[allowed],
            allowed_write_paths=[allowed],
            command_allowlist=["echo"],
        )
    )
    outside = tmp_path / "outside.txt"
    with pytest.raises(PermissionError, match="not allowed for write"):
        sandbox.execute(BashTool(), {"command": f"echo hello > {outside}"})


def test_security_command_tools_block_resource_exhaustion(tmp_path: Path, monkeypatch) -> None:
    _patch_shell_runner(monkeypatch)
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[tmp_path],
            allowed_write_paths=[tmp_path],
            command_denylist=["yes", "dd"],
        )
    )
    with pytest.raises(PermissionError, match="denylist"):
        sandbox.execute(BashTool(), {"command": "yes > /tmp/spam.txt"})
    with pytest.raises(PermissionError, match="denylist"):
        sandbox.execute(BashTool(), {"command": "dd if=/dev/zero of=/tmp/bigfile"})
