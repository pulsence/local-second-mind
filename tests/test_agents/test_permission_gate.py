from __future__ import annotations

from lsm.agents.permission_gate import PermissionGate
from lsm.agents.tools.base import BaseTool
from lsm.config.models.agents import SandboxConfig


class ReadTool(BaseTool):
    name = "read_file"
    description = "Read tool."
    risk_level = "read_only"

    def execute(self, args: dict) -> str:
        return "ok"


class WriteTool(BaseTool):
    name = "write_file"
    description = "Write tool."
    risk_level = "writes_workspace"
    requires_permission = True

    def execute(self, args: dict) -> str:
        return "ok"


class NetworkTool(BaseTool):
    name = "query_arxiv"
    description = "Network tool."
    risk_level = "network"
    needs_network = True

    def execute(self, args: dict) -> str:
        return "ok"


def test_permission_gate_tool_override_has_highest_precedence() -> None:
    gate = PermissionGate(
        SandboxConfig(
            require_user_permission={"write_file": False},
            require_permission_by_risk={"writes_workspace": True},
        )
    )
    decision = gate.check(WriteTool())
    assert decision.allowed is True
    assert decision.requires_confirmation is False


def test_permission_gate_risk_policy_applies_without_tool_override() -> None:
    gate = PermissionGate(SandboxConfig(require_permission_by_risk={"network": True}))
    decision = gate.check(NetworkTool())
    assert decision.allowed is False
    assert decision.requires_confirmation is True
    assert "Risk level 'network'" in decision.reason


def test_permission_gate_tool_default_applies_when_not_overridden() -> None:
    gate = PermissionGate(SandboxConfig())
    decision = gate.check(WriteTool())
    assert decision.allowed is False
    assert decision.requires_confirmation is True
    assert "requires user permission by default" in decision.reason


def test_permission_gate_allows_read_only_tools_by_default() -> None:
    gate = PermissionGate(SandboxConfig())
    decision = gate.check(ReadTool())
    assert decision.allowed is True
    assert decision.requires_confirmation is False
