from __future__ import annotations

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
    requires_permission = True

    def execute(self, args: dict) -> str:
        return "ok"


class LoadURLLikeTool(BaseTool):
    name = "load_url"
    description = "url"
    input_schema = {"type": "object", "properties": {"url": {"type": "string"}}}
    risk_level = "network"
    needs_network = True

    def execute(self, args: dict) -> str:
        return "ok"


def test_sandbox_allows_read_within_allowed_path(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[allowed],
            allowed_write_paths=[allowed],
        )
    )
    tool = ReadPathTool()
    result = sandbox.execute(tool, {"path": str(allowed / "a.txt")})
    assert result == "ok"


def test_sandbox_blocks_read_outside_allowed_path(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[allowed]))
    tool = ReadPathTool()
    with pytest.raises(PermissionError, match="not allowed for read"):
        sandbox.execute(tool, {"path": str(outside / "a.txt")})


def test_sandbox_blocks_write_when_user_permission_required(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_write_paths=[allowed],
            require_user_permission={"write_file": True},
        )
    )
    tool = WritePathTool()
    with pytest.raises(PermissionError, match="requires user permission"):
        sandbox.execute(tool, {"path": str(allowed / "out.txt")})


def test_sandbox_blocks_url_access_when_disabled(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    sandbox = ToolSandbox(
        SandboxConfig(
            allowed_read_paths=[allowed],
            allow_url_access=False,
        )
    )
    tool = LoadURLLikeTool()
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(tool, {"url": "https://example.com"})


def test_sandbox_rejects_local_paths_outside_global() -> None:
    global_root = Path.cwd()
    outside_root = Path(global_root.anchor) / "tmp"
    with pytest.raises(ValueError, match="outside global sandbox"):
        ToolSandbox(
            SandboxConfig(allowed_read_paths=[outside_root]),
            global_sandbox=SandboxConfig(allowed_read_paths=[global_root]),
        )
