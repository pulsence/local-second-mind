"""
Sandbox wrapper for agent tool execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional
from uuid import uuid4

from lsm.agents.interaction import InteractionChannel, InteractionRequest
from lsm.agents.log_redactor import redact_secrets
from lsm.agents.permission_gate import PermissionDecision, PermissionGate
from lsm.config.models.agents import SandboxConfig
from lsm.logging import get_logger
from lsm.utils.paths import resolve_path

from .env_scrubber import scrub_environment
from .base import BaseTool
from .docker_runner import DockerRunner
from .runner import BaseRunner, LocalRunner, ToolExecutionResult

logger = get_logger(__name__)


class ToolSandbox:
    """
    Enforce tool execution restrictions based on sandbox configuration.
    """

    _READ_TOOL_NAMES = {"read_file", "read_folder"}
    _WRITE_TOOL_NAMES = {"write_file", "create_folder"}
    _NETWORK_TOOL_NAMES = {"load_url", "query_llm", "query_remote", "query_remote_chain"}
    _DOCKER_ELIGIBLE_RISKS = {"network", "exec"}

    def __init__(
        self,
        config: SandboxConfig,
        global_sandbox: Optional[SandboxConfig] = None,
        *,
        local_runner: Optional[BaseRunner] = None,
        docker_runner: Optional[BaseRunner] = None,
        interaction_channel: Optional[InteractionChannel] = None,
        waiting_state_callback: Optional[Callable[[bool], None]] = None,
        workspace_root: Optional[Path | str] = None,
    ) -> None:
        self.config = config
        self.global_sandbox = global_sandbox
        self.permission_gate = PermissionGate(config)
        self.local_runner: BaseRunner = local_runner or self._build_local_runner()
        self.docker_runner: Optional[BaseRunner] = docker_runner or self._build_docker_runner()
        self.interaction_channel = interaction_channel
        self.waiting_state_callback = waiting_state_callback
        self.last_execution_result: Optional[ToolExecutionResult] = None
        self.workspace_root: Optional[Path] = (
            resolve_path(workspace_root, strict=False) if workspace_root is not None else None
        )
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
        self.last_execution_result = None
        resolved_args = self._resolve_workspace_args(tool, args)
        self._enforce_tool_permissions(tool, resolved_args)
        self._enforce_args(tool, resolved_args)
        runner = self._select_runner(tool)
        env = scrub_environment()
        result = runner.run(tool, resolved_args, env)
        result.stdout = redact_secrets(result.stdout)
        result.stderr = redact_secrets(result.stderr)
        self.last_execution_result = result
        return result.stdout

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

    def effective_read_paths(self) -> list[Path]:
        """
        Return read paths after applying global-sandbox restriction.
        """
        return list(self._effective_read_paths())

    def effective_write_paths(self) -> list[Path]:
        """
        Return write paths after applying global-sandbox restriction.
        """
        return list(self._effective_write_paths())

    def _enforce_tool_permissions(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        decision = self.permission_gate.check(tool, args)
        if decision.requires_confirmation:
            self._handle_permission_confirmation(decision, args)
        elif not decision.allowed:
            raise PermissionError(decision.reason)
        if self._is_network_tool(tool) and not self._allow_url_access():
            raise PermissionError("Network access is disabled by sandbox policy")

    def set_interaction_channel(
        self,
        interaction_channel: Optional[InteractionChannel],
        *,
        waiting_state_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """
        Update interaction channel and optional waiting-state callback.
        """
        self.interaction_channel = interaction_channel
        self.waiting_state_callback = waiting_state_callback

    def set_workspace_root(self, workspace_root: Optional[Path | str]) -> None:
        """
        Update the workspace root used to resolve relative file paths.
        """
        if workspace_root is None:
            self.workspace_root = None
            return
        self.workspace_root = resolve_path(workspace_root, strict=False)

    def _handle_permission_confirmation(
        self,
        decision: PermissionDecision,
        args: Dict[str, Any],
    ) -> None:
        channel = self.interaction_channel
        if channel is None:
            raise PermissionError(decision.reason)

        tool_name = str(decision.tool_name or "").strip()
        if tool_name and channel.has_session_approval(tool_name):
            return

        request = InteractionRequest(
            request_id=f"perm-{uuid4().hex}",
            request_type="permission",
            tool_name=tool_name or None,
            risk_level=str(decision.risk_level or "").strip() or None,
            reason=str(decision.reason or "").strip() or None,
            args_summary=self._summarize_args(args),
            prompt=(
                f"Allow tool '{tool_name or 'unknown_tool'}' "
                f"(risk: {str(decision.risk_level or 'read_only')})?"
            ),
        )

        self._notify_waiting_state(waiting=True)
        try:
            response = channel.post_request(request)
        except RuntimeError as exc:
            raise PermissionError(str(exc)) from exc
        finally:
            self._notify_waiting_state(waiting=False)

        if response.decision == "approve":
            return
        if response.decision == "approve_session":
            if tool_name:
                channel.approve_for_session(tool_name)
            return

        denial_reason = str(response.user_message or "").strip()
        if not denial_reason:
            denial_reason = decision.reason
        raise PermissionError(denial_reason)

    def _notify_waiting_state(self, *, waiting: bool) -> None:
        callback = self.waiting_state_callback
        if callback is None:
            return
        try:
            callback(waiting)
        except Exception:
            logger.exception("Waiting-state callback failed")

    @staticmethod
    def _summarize_args(args: Dict[str, Any]) -> str:
        try:
            text = json.dumps(args, ensure_ascii=True, sort_keys=True)
        except Exception:
            text = str(args)
        if len(text) > 500:
            return text[:497] + "..."
        return text

    def _enforce_args(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        self._validate_args_schema(tool, args)
        if tool.name in self._READ_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_read_path(Path(str(path)))
        if tool.name in self._WRITE_TOOL_NAMES:
            path = args.get("path")
            if path is not None:
                self.check_write_path(Path(str(path)))

    def _resolve_workspace_args(self, tool: BaseTool, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool.name not in self._READ_TOOL_NAMES and tool.name not in self._WRITE_TOOL_NAMES:
            return args
        if self.workspace_root is None:
            return args
        path_value = args.get("path")
        if path_value is None:
            return args
        candidate = Path(str(path_value)).expanduser()
        if candidate.is_absolute():
            return args
        resolved = resolve_path(candidate, base_dir=self.workspace_root, strict=False)
        updated = dict(args)
        updated["path"] = str(resolved)
        return updated

    def _validate_args_schema(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")

        schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                key_name = str(key)
                if key_name not in args:
                    raise ValueError(
                        f"Missing required argument '{key_name}' for tool '{tool.name}'"
                    )

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return

        unexpected = sorted(set(args.keys()) - set(properties.keys()))
        if unexpected:
            raise ValueError(
                f"Unexpected argument(s) for tool '{tool.name}': {', '.join(unexpected)}"
            )

        for key, value in args.items():
            expected_type = properties.get(key, {}).get("type")
            if expected_type is None:
                continue
            if not self._matches_json_type(value, str(expected_type)):
                actual_type = type(value).__name__
                raise ValueError(
                    f"Argument '{key}' for tool '{tool.name}' must be type '{expected_type}' (got {actual_type})"
                )

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
        if self.config.allow_url_access and not self.global_sandbox.allow_url_access:
            raise ValueError(
                "sandbox.allow_url_access cannot be enabled when disabled in global sandbox"
            )
        if self.global_sandbox.force_docker and not self.config.force_docker:
            raise ValueError(
                "sandbox.force_docker cannot be disabled relative to global sandbox"
            )
        if (
            self.global_sandbox.execution_mode == "prefer_docker"
            and self.config.execution_mode != "prefer_docker"
        ):
            raise ValueError(
                "sandbox.execution_mode cannot be relaxed from global 'prefer_docker'"
            )
        self._validate_permission_subset(
            self.config.require_user_permission,
            self.global_sandbox.require_user_permission,
            "require_user_permission",
        )
        self._validate_permission_subset(
            self.config.require_permission_by_risk,
            self.global_sandbox.require_permission_by_risk,
            "require_permission_by_risk",
        )
        self._validate_limits_subset(self.config.limits, self.global_sandbox.limits)

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

    def _validate_permission_subset(
        self,
        local_flags: Dict[str, bool],
        global_flags: Dict[str, bool],
        field_name: str,
    ) -> None:
        for key, parent_requires in global_flags.items():
            if not bool(parent_requires):
                continue
            if not bool(local_flags.get(key, False)):
                raise ValueError(
                    f"sandbox.{field_name}['{key}'] cannot be disabled relative to global sandbox"
                )

    def _validate_limits_subset(
        self,
        local_limits: Dict[str, Any],
        global_limits: Dict[str, Any],
    ) -> None:
        try:
            local_timeout = float(local_limits.get("timeout_s_default", 30.0))
            global_timeout = float(global_limits.get("timeout_s_default", 30.0))
            local_stdout = int(local_limits.get("max_stdout_kb", 256))
            global_stdout = int(global_limits.get("max_stdout_kb", 256))
            local_write_mb = float(local_limits.get("max_file_write_mb", 10.0))
            global_write_mb = float(global_limits.get("max_file_write_mb", 10.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("sandbox.limits values must be numeric") from exc

        if local_timeout > global_timeout:
            raise ValueError(
                "sandbox.limits.timeout_s_default cannot exceed global sandbox limit"
            )
        if local_stdout > global_stdout:
            raise ValueError(
                "sandbox.limits.max_stdout_kb cannot exceed global sandbox limit"
            )
        if local_write_mb > global_write_mb:
            raise ValueError(
                "sandbox.limits.max_file_write_mb cannot exceed global sandbox limit"
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

    @staticmethod
    def _matches_json_type(value: Any, expected_type: str) -> bool:
        if expected_type == "string":
            return isinstance(value, str)
        if expected_type == "boolean":
            return isinstance(value, bool)
        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected_type == "number":
            return (
                isinstance(value, int) and not isinstance(value, bool)
            ) or isinstance(value, float)
        if expected_type == "object":
            return isinstance(value, dict)
        if expected_type == "array":
            return isinstance(value, list)
        if expected_type == "null":
            return value is None
        return True

    def _is_network_tool(self, tool: BaseTool) -> bool:
        return (
            tool.needs_network
            or tool.risk_level == "network"
            or tool.name in self._NETWORK_TOOL_NAMES
        )

    def _build_local_runner(self) -> BaseRunner:
        return LocalRunner(
            timeout_s_default=float(self.config.limits.get("timeout_s_default", 30.0)),
            max_stdout_kb=int(self.config.limits.get("max_stdout_kb", 256)),
            max_file_write_mb=float(self.config.limits.get("max_file_write_mb", 10.0)),
        )

    def _build_docker_runner(self) -> Optional[BaseRunner]:
        if not bool(self.config.docker.get("enabled", False)):
            return None
        return DockerRunner(
            image=str(self.config.docker.get("image", "lsm-agent-sandbox:latest")),
            workspace_root=Path.cwd(),
            read_paths=self._effective_read_paths(),
            timeout_s_default=float(self.config.limits.get("timeout_s_default", 30.0)),
            max_stdout_kb=int(self.config.limits.get("max_stdout_kb", 256)),
            network_default=str(self.config.docker.get("network_default", "none")),
            cpu_limit=float(self.config.docker.get("cpu_limit", 1.0)),
            mem_limit_mb=int(self.config.docker.get("mem_limit_mb", 512)),
            read_only_root=bool(self.config.docker.get("read_only_root", True)),
        )

    def _select_runner(self, tool: BaseTool) -> BaseRunner:
        risk_level = str(tool.risk_level or "read_only")
        mode = self.config.execution_mode

        if self.config.force_docker:
            if bool(self.config.docker.get("enabled", False)) and self.docker_runner is not None:
                logger.debug(
                    "Runner selection: tool='%s' risk='%s' mode='%s' force_docker=true runner='docker'",
                    tool.name,
                    risk_level,
                    mode,
                )
                return self.docker_runner
            reason = (
                f"Tool '{tool.name}' requires Docker execution because sandbox.force_docker=true, "
                "but Docker runner is unavailable"
            )
            logger.warning(
                "Runner selection blocked: tool='%s' risk='%s' mode='%s' reason='%s'",
                tool.name,
                risk_level,
                mode,
                reason,
            )
            raise PermissionError(reason)

        if risk_level in {"read_only", "writes_workspace"}:
            logger.debug(
                "Runner selection: tool='%s' risk='%s' mode='%s' runner='local' reason='low-risk policy'",
                tool.name,
                risk_level,
                mode,
            )
            return self.local_runner

        if risk_level in self._DOCKER_ELIGIBLE_RISKS:
            if mode == "prefer_docker":
                if bool(self.config.docker.get("enabled", False)) and self.docker_runner is not None:
                    logger.debug(
                        "Runner selection: tool='%s' risk='%s' mode='%s' runner='docker' reason='prefer_docker high-risk policy'",
                        tool.name,
                        risk_level,
                        mode,
                    )
                    return self.docker_runner
                reason = (
                    f"Tool '{tool.name}' (risk '{risk_level}') requires user confirmation "
                    "for local execution because Docker runner is unavailable"
                )
                logger.warning(
                    "Runner selection blocked: tool='%s' risk='%s' mode='%s' reason='%s'",
                    tool.name,
                    risk_level,
                    mode,
                    reason,
                )
                raise PermissionError(reason)

            logger.debug(
                "Runner selection: tool='%s' risk='%s' mode='%s' runner='local' reason='execution_mode local_only'",
                tool.name,
                risk_level,
                mode,
            )
            return self.local_runner

        logger.debug(
            "Runner selection: tool='%s' risk='%s' mode='%s' runner='local' reason='fallback'",
            tool.name,
            risk_level,
            mode,
        )
        return self.local_runner

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False
