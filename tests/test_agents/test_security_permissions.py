from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.models.agents import SandboxConfig


class StrictReadTool(BaseTool):
    name = "read_file"
    description = "strict read"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        return "ok"


class WriteTool(BaseTool):
    name = "write_file"
    description = "write"
    risk_level = "writes_workspace"
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        return "ok"


class NetworkTool(BaseTool):
    name = "query_remote"
    description = "network"
    risk_level = "network"
    needs_network = True
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> str:
        _ = args
        return "ok"


def test_security_unknown_tool_name_rejected_by_registry() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.lookup("unknown_tool")


def test_security_tool_schema_rejects_extra_fields(tmp_path: Path) -> None:
    sandbox = ToolSandbox(SandboxConfig(allowed_read_paths=[tmp_path]))
    tool = StrictReadTool()
    with pytest.raises(ValueError, match="Unexpected argument"):
        sandbox.execute(
            tool,
            {"path": str(tmp_path / "note.txt"), "unexpected": "boom"},
        )


def test_security_permission_gate_blocks_tools_requiring_permission(tmp_path: Path) -> None:
    sandbox = ToolSandbox(SandboxConfig(allowed_write_paths=[tmp_path]))
    tool = WriteTool()
    with pytest.raises(PermissionError, match="requires user permission"):
        sandbox.execute(tool, {"path": str(tmp_path / "out.txt")})


def test_security_per_risk_policy_is_enforced(tmp_path: Path) -> None:
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            require_permission_by_risk={"network": True},
        )
    )
    tool = NetworkTool()
    with pytest.raises(PermissionError, match="Risk level 'network'"):
        sandbox.execute(tool, {})


def test_security_per_tool_override_wins_over_risk_policy(tmp_path: Path) -> None:
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=True,
            require_user_permission={"query_remote": False},
            require_permission_by_risk={"network": True},
        )
    )
    tool = NetworkTool()
    assert sandbox.execute(tool, {}) == "ok"


def test_security_write_tools_blocked_when_no_write_paths(tmp_path: Path) -> None:
    sandbox = ToolSandbox(
        SandboxConfig(
            require_user_permission={"write_file": False},
        )
    )
    tool = WriteTool()
    with pytest.raises(PermissionError, match="No write paths are allowed"):
        sandbox.execute(tool, {"path": str(tmp_path / "out.txt")})


def test_security_network_tools_blocked_when_url_access_disabled() -> None:
    sandbox = ToolSandbox(
        SandboxConfig(
            allow_url_access=False,
            require_user_permission={"query_remote": False},
        )
    )
    tool = NetworkTool()
    with pytest.raises(PermissionError, match="Network access is disabled"):
        sandbox.execute(tool, {})


def test_security_local_sandbox_cannot_exceed_global_scope(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    outside_root = tmp_path / "outside"
    global_root.mkdir()
    outside_root.mkdir()

    with pytest.raises(ValueError, match="outside global sandbox"):
        ToolSandbox(
            SandboxConfig(allowed_read_paths=[outside_root]),
            global_sandbox=SandboxConfig(allowed_read_paths=[global_root]),
        )
