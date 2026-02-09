"""
Permission decision model and policy evaluation for agent tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from lsm.config.models.agents import SandboxConfig

from .tools.base import BaseTool


@dataclass(frozen=True)
class PermissionDecision:
    """
    Decision returned by the permission gate.
    """

    allowed: bool
    reason: str
    requires_confirmation: bool
    tool_name: str
    risk_level: str


class PermissionGate:
    """
    Evaluate permission policies for tool execution.
    """

    def __init__(self, sandbox_config: SandboxConfig) -> None:
        self.sandbox_config = sandbox_config

    def check(self, tool: BaseTool, args: Optional[Dict[str, Any]] = None) -> PermissionDecision:
        """
        Evaluate whether a tool may execute immediately.

        Precedence:
        1. Explicit per-tool override (`require_user_permission`)
        2. Per-risk policy (`require_permission_by_risk`)
        3. Tool default (`requires_permission`)
        4. Allow

        Args:
            tool: Tool under evaluation.
            args: Tool arguments (reserved for future use).

        Returns:
            PermissionDecision describing whether execution is allowed.
        """
        _ = args
        tool_name = str(tool.name).strip()
        risk_level = str(tool.risk_level).strip() or "read_only"

        per_tool = self._get_per_tool_policy(tool_name)
        if per_tool is not None:
            if per_tool:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Tool '{tool_name}' requires user permission",
                    requires_confirmation=True,
                    tool_name=tool_name,
                    risk_level=risk_level,
                )
            return PermissionDecision(
                allowed=True,
                reason=f"Tool '{tool_name}' is explicitly allowed by sandbox policy",
                requires_confirmation=False,
                tool_name=tool_name,
                risk_level=risk_level,
            )

        per_risk = self._get_per_risk_policy(risk_level)
        if per_risk is not None:
            if per_risk:
                return PermissionDecision(
                    allowed=False,
                    reason=f"Risk level '{risk_level}' requires user permission",
                    requires_confirmation=True,
                    tool_name=tool_name,
                    risk_level=risk_level,
                )
            return PermissionDecision(
                allowed=True,
                reason=f"Risk level '{risk_level}' is allowed by sandbox policy",
                requires_confirmation=False,
                tool_name=tool_name,
                risk_level=risk_level,
            )

        if bool(tool.requires_permission):
            return PermissionDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' requires user permission by default",
                requires_confirmation=True,
                tool_name=tool_name,
                risk_level=risk_level,
            )

        return PermissionDecision(
            allowed=True,
            reason=f"Tool '{tool_name}' allowed",
            requires_confirmation=False,
            tool_name=tool_name,
            risk_level=risk_level,
        )

    def _get_per_tool_policy(self, tool_name: str) -> Optional[bool]:
        if tool_name not in self.sandbox_config.require_user_permission:
            return None
        return bool(self.sandbox_config.require_user_permission[tool_name])

    def _get_per_risk_policy(self, risk_level: str) -> Optional[bool]:
        if risk_level not in self.sandbox_config.require_permission_by_risk:
            return None
        return bool(self.sandbox_config.require_permission_by_risk[risk_level])
