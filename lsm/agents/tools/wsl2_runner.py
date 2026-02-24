"""
WSL2 runner for sandboxed tool execution.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Mapping
import shlex

from lsm.logging import get_logger

from .base import BaseTool
from .env_scrubber import scrub_environment
from .runner import BaseRunner, ToolExecutionResult

logger = get_logger(__name__)


class WSL2Runner(BaseRunner):
    """
    Execute tools inside a WSL2 environment.
    """

    _ENTRYPOINT_MODULE = "lsm.agents.tools._docker_entrypoint"

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        distro: str | None = None,
        wsl_bin: str = "wsl",
        shell: str = "bash",
        timeout_s_default: float = 30.0,
        max_stdout_kb: int = 256,
    ) -> None:
        self.workspace_root = str(workspace_root) if workspace_root is not None else None
        self.distro = str(distro or "").strip()
        self.wsl_bin = str(wsl_bin or "wsl").strip()
        self.shell = str(shell or "bash").strip()
        self.timeout_s_default = float(timeout_s_default)
        self.max_stdout_kb = int(max_stdout_kb)

    def is_available(self) -> bool:
        return shutil.which(self.wsl_bin) is not None

    def run(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        env: Mapping[str, str],
    ) -> ToolExecutionResult:
        if not self.is_available():
            raise RuntimeError("WSL2 CLI is not available on PATH")

        clean_env = scrub_environment(env)
        translated_args = self._translate_args(args)
        payload = {
            "tool_name": tool.name,
            "tool_module": tool.__class__.__module__,
            "tool_class": tool.__class__.__name__,
            "args": translated_args,
        }
        command = self._build_command(clean_env)
        logger.debug("WSL2 runner executing tool='%s' distro='%s'", tool.name, self.distro or "default")
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=max(0.001, self.timeout_s_default),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Tool '{tool.name}' exceeded timeout of {self.timeout_s_default:.3f}s in WSL2 runner"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"WSL2 runner failed to start: {exc}") from exc

        runtime_ms = (time.perf_counter() - started) * 1000.0

        if proc.returncode != 0:
            stderr = str(proc.stderr or "").strip()
            message = stderr or f"wsl exited with status {proc.returncode}"
            logger.warning(
                "WSL2 runner failed tool='%s' distro='%s' error='%s'",
                tool.name,
                self.distro or "default",
                message,
            )
            raise RuntimeError(f"WSL2 runner failed for tool '{tool.name}': {message}")

        stdout, stderr, artifacts = self._decode_response(proc.stdout, proc.stderr)
        return ToolExecutionResult(
            stdout=self._truncate_text(stdout),
            stderr=self._truncate_text(stderr),
            runner_used="wsl2",
            runtime_ms=runtime_ms,
            artifacts=artifacts,
        )

    def _build_command(self, env: Mapping[str, str]) -> List[str]:
        command: List[str] = [self.wsl_bin]
        if self.distro:
            command.extend(["-d", self.distro])
        command.append("--")
        command.append(self.shell)
        command.append("-lc")

        script_parts: list[str] = []
        if self.workspace_root:
            workspace_wsl = self._to_wsl_path(self.workspace_root)
            script_parts.append(f"cd {shlex.quote(workspace_wsl)}")

        env_items = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
        python_cmd = f"python -m {self._ENTRYPOINT_MODULE}"
        if env_items:
            python_cmd = f"env {' '.join(env_items)} {python_cmd}"
        script_parts.append(python_cmd)
        command.append(" && ".join(script_parts))
        return command

    def _decode_response(
        self,
        stdout: str,
        stderr: str,
    ) -> tuple[str, str, list[str]]:
        text = str(stdout or "").strip()
        if not text:
            return "", str(stderr or ""), []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text, str(stderr or ""), []
        if not isinstance(parsed, dict):
            return text, str(stderr or ""), []

        parsed_stdout = str(parsed.get("stdout", ""))
        parsed_stderr = str(parsed.get("stderr", ""))
        merged_stderr = str(stderr or "")
        if parsed_stderr:
            merged_stderr = f"{merged_stderr}\n{parsed_stderr}".strip()
        artifacts_raw = parsed.get("artifacts", [])
        artifacts = []
        if isinstance(artifacts_raw, list):
            artifacts = [str(item) for item in artifacts_raw if str(item).strip()]
        return parsed_stdout, merged_stderr, artifacts

    def _truncate_text(self, text: str) -> str:
        max_bytes = max(1, self.max_stdout_kb) * 1024
        encoded = str(text).encode("utf-8")
        if len(encoded) <= max_bytes:
            return str(text)
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        removed = len(encoded) - max_bytes
        return f"{truncated}\n[TRUNCATED {removed} bytes]"

    def _translate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        def translate(value: Any) -> Any:
            if isinstance(value, str):
                return self._to_wsl_path(value)
            if isinstance(value, list):
                return [translate(item) for item in value]
            if isinstance(value, dict):
                return {key: translate(item) for key, item in value.items()}
            return value

        return translate(dict(args))

    def _to_wsl_path(self, value: str) -> str:
        text = str(value).strip()
        if not text:
            return value

        if self._looks_like_windows_path(text):
            win = PureWindowsPath(text)
            drive = win.drive.rstrip(":").lower()
            tail = "/".join(win.parts[1:])
            if not drive:
                return text
            if tail:
                return f"/mnt/{drive}/{tail}"
            return f"/mnt/{drive}"

        path = Path(text).expanduser()
        if path.is_absolute():
            return path.as_posix()
        return text

    @staticmethod
    def _looks_like_windows_path(text: str) -> bool:
        if len(text) < 2:
            return False
        if text[1] == ":" and text[0].isalpha():
            return True
        if text.startswith("\\\\"):
            return True
        return False
