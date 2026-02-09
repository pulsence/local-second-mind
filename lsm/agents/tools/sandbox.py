"""
Sandbox wrapper for agent tool execution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from lsm.agents.permission_gate import PermissionGate
from lsm.config.models.agents import SandboxConfig

from .base import BaseTool


class ToolSandbox:
    """
    Enforce tool execution restrictions based on sandbox configuration.
    """

    _READ_TOOL_NAMES = {"read_file", "read_folder"}
    _WRITE_TOOL_NAMES = {"write_file", "create_folder"}
    _NETWORK_TOOL_NAMES = {"load_url", "query_llm", "query_remote", "query_remote_chain"}

    def __init__(
        self,
        config: SandboxConfig,
        global_sandbox: Optional[SandboxConfig] = None,
    ) -> None:
        self.config = config
        self.global_sandbox = global_sandbox
        self.permission_gate = PermissionGate(config)
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
        decision = self.permission_gate.check(tool, {})
        if decision.requires_confirmation:
            raise PermissionError(decision.reason)
        if not decision.allowed:
            raise PermissionError(decision.reason)
        if self._is_network_tool(tool) and not self._allow_url_access():
            raise PermissionError("Network access is disabled by sandbox policy")

    def _enforce_args(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        if tool.name in self._READ_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_read_path(Path(str(path)))
        if tool.name in self._WRITE_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_write_path(Path(str(path)))

    def _check_path(self, path: Path, allowed_paths: Iterable[Path], action: str) -> None:
        resolved = self._canonicalize_path(path)
        allowed = [self._canonicalize_path(root) for root in allowed_paths]
        if not allowed:
            raise PermissionError(f"No {action} paths are allowed in sandbox")
        self._reject_symlink_escape(path, allowed, action)
        if not any(self._is_relative_to(resolved, root) for root in allowed):
            raise PermissionError(f"Path '{resolved}' is not allowed for {action}")

    def _allow_url_access(self) -> bool:
        if self.global_sandbox is None:
            return self.config.allow_url_access
        return self.config.allow_url_access and self.global_sandbox.allow_url_access

    def _effective_read_paths(self) -> list[Path]:
        local_paths = [self._canonicalize_path(path) for path in self.config.allowed_read_paths]
        if self.global_sandbox is None:
            return local_paths
        global_paths = [
            self._canonicalize_path(path) for path in self.global_sandbox.allowed_read_paths
        ]
        return self._restrict_to_global(local_paths, global_paths)

    def _effective_write_paths(self) -> list[Path]:
        local_paths = [self._canonicalize_path(path) for path in self.config.allowed_write_paths]
        if self.global_sandbox is None:
            return local_paths
        global_paths = [
            self._canonicalize_path(path) for path in self.global_sandbox.allowed_write_paths
        ]
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
        global_resolved = [self._canonicalize_path(path) for path in global_paths]
        for local_path in local_paths:
            local_resolved = self._canonicalize_path(local_path)
            if not any(self._is_relative_to(local_resolved, global_path) for global_path in global_resolved):
                raise ValueError(
                    f"sandbox.{field_name} includes path outside global sandbox: {local_resolved}"
                )

    def _canonicalize_path(self, path: Path) -> Path:
        """
        Normalize and validate an input path.

        Args:
            path: Candidate path.

        Returns:
            Canonical absolute path.

        Raises:
            ValueError: For malformed paths.
            PermissionError: For prohibited path forms.
        """
        raw = str(path)
        if not raw.strip():
            raise ValueError("Path cannot be empty")
        if "\x00" in raw:
            raise ValueError("Path contains null bytes")
        if os.name == "nt":
            if raw.startswith("\\\\"):
                raise PermissionError("UNC paths are not allowed")
            if self._has_windows_ads(path):
                raise PermissionError("Alternate data streams are not allowed")

        expanded = path.expanduser()
        if any(part == ".." for part in expanded.parts):
            raise PermissionError("Path traversal components are not allowed")

        resolved = expanded.resolve(strict=False)
        if any(part == ".." for part in resolved.parts):
            raise PermissionError("Path traversal components are not allowed")
        return resolved

    def _reject_symlink_escape(
        self,
        path: Path,
        allowed_paths: list[Path],
        action: str,
    ) -> None:
        """
        Reject symlinked path components that resolve outside allowed roots.
        """
        candidate = path.expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate

        parts = candidate.parts
        if not parts:
            return

        current = Path(parts[0])
        for part in parts[1:]:
            current = current / part
            if not current.exists() or not current.is_symlink():
                continue
            target = current.resolve(strict=False)
            if not any(self._is_relative_to(target, root) for root in allowed_paths):
                raise PermissionError(
                    f"Symlink path '{current}' resolves outside allowed {action} paths"
                )

    @staticmethod
    def _has_windows_ads(path: Path) -> bool:
        for part in path.parts:
            normalized = part.replace("\\", "").replace("/", "")
            if len(normalized) == 2 and normalized[1] == ":" and normalized[0].isalpha():
                continue
            if ":" in part:
                return True
        return False

    def _is_network_tool(self, tool: BaseTool) -> bool:
        return (
            tool.needs_network
            or tool.risk_level == "network"
            or tool.name in self._NETWORK_TOOL_NAMES
        )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
