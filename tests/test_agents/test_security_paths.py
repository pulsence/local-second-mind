from __future__ import annotations

import os
from pathlib import Path

import pytest

from lsm.agents.tools.base import BaseTool
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


class ReadPathTool(BaseTool):
    name = "read_file"
    description = "read"
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, args: dict) -> str:
        return "ok"


class WritePathTool(BaseTool):
    name = "write_file"
    description = "write"
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    requires_permission = False
    risk_level = "writes_workspace"

    def execute(self, args: dict) -> str:
        return "ok"


def test_sandbox_rejects_null_byte_in_path(tmp_path: Path) -> None:
    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[tmp_path]))
    tool = ReadPathTool()
    with pytest.raises(ValueError, match="null bytes"):
        sandbox.execute(tool, {"path": str(tmp_path / "a\x00.txt")})


def test_sandbox_rejects_dot_dot_path_traversal(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[allowed]))
    tool = ReadPathTool()
    with pytest.raises(PermissionError, match="Path traversal"):
        sandbox.execute(tool, {"path": str(allowed / ".." / "outside.txt")})


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific path rule")
def test_sandbox_rejects_unc_path_on_windows(tmp_path: Path) -> None:
    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[tmp_path]))
    tool = ReadPathTool()
    with pytest.raises(PermissionError, match="UNC paths are not allowed"):
        sandbox.execute(tool, {"path": r"\\server\share\file.txt"})


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific path rule")
def test_sandbox_rejects_alternate_data_stream_on_windows(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(SandboxConfig(allowed_write_paths=[allowed]))
    tool = WritePathTool()
    with pytest.raises(PermissionError, match="Alternate data streams"):
        sandbox.execute(tool, {"path": str(allowed / "note.txt:secret")})


def test_sandbox_blocks_symlink_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    symlink_path = allowed / "secret_link.txt"
    try:
        symlink_path.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Symlink creation not permitted in this environment")

    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[allowed]))
    tool = ReadPathTool()
    with pytest.raises(PermissionError, match="Symlink path"):
        sandbox.execute(tool, {"path": str(symlink_path)})
