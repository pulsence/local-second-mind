"""
Sandbox wrapper for agent tool execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from lsm.config.models.agents import SandboxConfig

from .base import BaseTool


class ToolSandbox:
    """
    Enforce tool execution restrictions based on sandbox configuration.
    """

    _READ_TOOL_NAMES = {"read_file", "read_folder"}
    _WRITE_TOOL_NAMES = {"write_file", "create_folder"}

    def __init__(
        self,
        config: SandboxConfig,
        global_sandbox: Optional[SandboxConfig] = None,
    ) -> None:
        self.config = config
        self.global_sandbox = global_sandbox
        self._validate_not_exceeding_global()

    def execute(self, tool: BaseTool, args: Dict[str, Any]) -> str:
        """
        Execute a tool after sandbox checks.

        Args:
            tool: Tool instance to execute.
            args: Tool arguments.

        Returns:
            Tool output text.
        """
        self._enforce_tool_permissions(tool)
        self._enforce_args(tool, args)
        return tool.execute(args)

    def check_read_path(self, path: Path) -> None:
        """
        Validate a path against read allowlist.

        Args:
            path: Path to validate.
        """
        self._check_path(path, self._effective_read_paths(), "read")

    def check_write_path(self, path: Path) -> None:
        """
        Validate a path against write allowlist.

        Args:
            path: Path to validate.
        """
        self._check_path(path, self._effective_write_paths(), "write")

    def _enforce_tool_permissions(self, tool: BaseTool) -> None:
        if tool.requires_permission and self.config.require_user_permission.get(tool.name, False):
            raise PermissionError(f"Tool '{tool.name}' requires user permission")
        if tool.name == "load_url" and not self._allow_url_access():
            raise PermissionError("URL access is disabled by sandbox policy")

    def _enforce_args(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        if tool.name in self._READ_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_read_path(Path(path))
        if tool.name in self._WRITE_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_write_path(Path(path))

    def _check_path(self, path: Path, allowed_paths: Iterable[Path], action: str) -> None:
        resolved = path.expanduser().resolve()
        allowed = list(allowed_paths)
        if not allowed:
            raise PermissionError(f"No {action} paths are allowed in sandbox")
        if not any(self._is_relative_to(resolved, root) for root in allowed):
            raise PermissionError(f"Path '{resolved}' is not allowed for {action}")

    def _allow_url_access(self) -> bool:
        if self.global_sandbox is None:
            return self.config.allow_url_access
        return self.config.allow_url_access and self.global_sandbox.allow_url_access

    def _effective_read_paths(self) -> list[Path]:
        local_paths = [path.resolve() for path in self.config.allowed_read_paths]
        if self.global_sandbox is None:
            return local_paths
        global_paths = [path.resolve() for path in self.global_sandbox.allowed_read_paths]
        return self._restrict_to_global(local_paths, global_paths)

    def _effective_write_paths(self) -> list[Path]:
        local_paths = [path.resolve() for path in self.config.allowed_write_paths]
        if self.global_sandbox is None:
            return local_paths
        global_paths = [path.resolve() for path in self.global_sandbox.allowed_write_paths]
        return self._restrict_to_global(local_paths, global_paths)

    def _restrict_to_global(self, local_paths: list[Path], global_paths: list[Path]) -> list[Path]:
        if not global_paths:
            return []
        if not local_paths:
            return []
        effective: list[Path] = []
        for local_path in local_paths:
            if any(self._is_relative_to(local_path, global_path) for global_path in global_paths):
                effective.append(local_path)
        return effective

    def _validate_not_exceeding_global(self) -> None:
        if self.global_sandbox is None:
            return
        self._validate_subset(
            self.config.allowed_read_paths,
            self.global_sandbox.allowed_read_paths,
            "allowed_read_paths",
        )
        self._validate_subset(
            self.config.allowed_write_paths,
            self.global_sandbox.allowed_write_paths,
            "allowed_write_paths",
        )

    def _validate_subset(
        self,
        local_paths: list[Path],
        global_paths: list[Path],
        field_name: str,
    ) -> None:
        global_resolved = [path.resolve() for path in global_paths]
        for local_path in local_paths:
            local_resolved = local_path.resolve()
            if not any(self._is_relative_to(local_resolved, global_path) for global_path in global_resolved):
                raise ValueError(
                    f"sandbox.{field_name} includes path outside global sandbox: {local_resolved}"
                )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

