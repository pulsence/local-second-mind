"""
Runner abstraction for sandboxed tool execution.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .base import BaseTool


@dataclass
class ToolExecutionResult:
    """
    Structured result for a tool execution.
    """

    stdout: str
    stderr: str = ""
    runner_used: str = "local"
    runtime_ms: float = 0.0
    artifacts: List[str] = field(default_factory=list)


class BaseRunner(ABC):
    """
    Abstract runner interface for tool execution.
    """

    @abstractmethod
    def run(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        env: Mapping[str, str],
    ) -> ToolExecutionResult:
        """
        Execute a tool in this runner.

        Args:
            tool: Tool instance to execute.
            args: Tool arguments.
            env: Sanitized environment map.

        Returns:
            Structured execution result.
        """


class LocalRunner(BaseRunner):
    """
    Local in-process tool runner with timeout and output limiting.
    """

    def __init__(
        self,
        *,
        timeout_s_default: float = 30.0,
        max_stdout_kb: int = 256,
        max_file_write_mb: float = 10.0,
    ) -> None:
        self.timeout_s_default = float(timeout_s_default)
        self.max_stdout_kb = int(max_stdout_kb)
        self.max_file_write_mb = float(max_file_write_mb)

    def run(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        env: Mapping[str, str],
    ) -> ToolExecutionResult:
        _ = env  # Reserved for parity with future isolated/subprocess runners.
        self._enforce_write_size_limit(tool, args)

        output_holder: dict[str, str] = {}
        error_holder: dict[str, BaseException] = {}

        def _execute() -> None:
            try:
                output_holder["stdout"] = str(tool.execute(args))
            except BaseException as exc:  # pragma: no cover - raised after join.
                error_holder["error"] = exc

        started = time.perf_counter()
        worker = threading.Thread(target=_execute, daemon=True)
        worker.start()
        worker.join(timeout=max(0.001, self.timeout_s_default))
        runtime_ms = (time.perf_counter() - started) * 1000.0

        if worker.is_alive():
            raise TimeoutError(
                f"Tool '{tool.name}' exceeded timeout of {self.timeout_s_default:.3f}s"
            )
        if "error" in error_holder:
            raise error_holder["error"]

        stdout = self._truncate_stdout(output_holder.get("stdout", ""))
        artifacts = self._collect_artifacts(tool, args)
        return ToolExecutionResult(
            stdout=stdout,
            stderr="",
            runner_used="local",
            runtime_ms=runtime_ms,
            artifacts=artifacts,
        )

    def _truncate_stdout(self, stdout: str) -> str:
        max_bytes = max(1, self.max_stdout_kb) * 1024
        encoded = stdout.encode("utf-8")
        if len(encoded) <= max_bytes:
            return stdout
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        removed = len(encoded) - max_bytes
        return f"{truncated}\n[TRUNCATED {removed} bytes]"

    def _collect_artifacts(self, tool: BaseTool, args: Dict[str, Any]) -> List[str]:
        if tool.risk_level != "writes_workspace":
            return []
        path_value = args.get("path")
        if path_value is None:
            return []
        path = Path(str(path_value).strip())
        if not str(path):
            return []
        return [str(path.expanduser().resolve(strict=False))]

    def _enforce_write_size_limit(self, tool: BaseTool, args: Dict[str, Any]) -> None:
        if tool.risk_level != "writes_workspace":
            return
        content = args.get("content")
        if content is None:
            return
        size_mb = len(str(content).encode("utf-8")) / (1024 * 1024)
        if size_mb > max(0.001, self.max_file_write_mb):
            raise ValueError(
                f"Tool '{tool.name}' content exceeds max_file_write_mb ({self.max_file_write_mb})"
            )
