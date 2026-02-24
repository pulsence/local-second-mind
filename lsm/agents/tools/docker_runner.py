"""
Docker runner foundation for sandboxed tool execution.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from lsm.logging import get_logger

from .base import BaseTool
from .env_scrubber import scrub_environment
from .runner import BaseRunner, ToolExecutionResult

logger = get_logger(__name__)


class DockerRunner(BaseRunner):
    """
    Execute tools inside a constrained Docker container.
    """

    _ENTRYPOINT_MODULE = "lsm.agents.tools._docker_entrypoint"

    def __init__(
        self,
        *,
        image: str = "lsm-agent-sandbox:latest",
        workspace_root: Path | str | None = None,
        read_paths: Sequence[Path | str] | None = None,
        write_paths: Sequence[Path | str] | None = None,
        timeout_s_default: float = 30.0,
        max_stdout_kb: int = 256,
        network_default: str = "none",
        cpu_limit: float = 1.0,
        mem_limit_mb: int = 512,
        read_only_root: bool = True,
        pids_limit: int = 256,
        docker_bin: str = "docker",
    ) -> None:
        self.image = str(image or "lsm-agent-sandbox:latest").strip()
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve(strict=False)
        self.read_paths = self._normalize_read_paths(read_paths)
        self.write_paths = self._normalize_write_paths(write_paths)
        self.timeout_s_default = float(timeout_s_default)
        self.max_stdout_kb = int(max_stdout_kb)
        self.network_default = str(network_default or "none").strip() or "none"
        self.cpu_limit = float(cpu_limit)
        self.mem_limit_mb = int(mem_limit_mb)
        self.read_only_root = bool(read_only_root)
        self.pids_limit = int(pids_limit)
        self.docker_bin = str(docker_bin or "docker")

    def run(
        self,
        tool: BaseTool,
        args: Dict[str, Any],
        env: Mapping[str, str],
    ) -> ToolExecutionResult:
        if shutil.which(self.docker_bin) is None:
            raise RuntimeError("Docker CLI is not available on PATH")

        clean_env = scrub_environment(env)
        translated_args = self._translate_args(args)
        payload = {
            "tool_name": tool.name,
            "tool_module": tool.__class__.__module__,
            "tool_class": tool.__class__.__name__,
            "args": translated_args,
        }
        command = self._build_command(clean_env)
        logger.debug(
            "Docker runner executing tool='%s' image='%s'",
            tool.name,
            self.image,
        )
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
                f"Tool '{tool.name}' exceeded timeout of {self.timeout_s_default:.3f}s in docker runner"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Docker runner failed to start: {exc}") from exc

        runtime_ms = (time.perf_counter() - started) * 1000.0

        if proc.returncode != 0:
            stderr = str(proc.stderr or "").strip()
            message = stderr or f"docker exited with status {proc.returncode}"
            logger.warning(
                "Docker runner failed tool='%s' image='%s' error='%s'",
                tool.name,
                self.image,
                message,
            )
            raise RuntimeError(f"Docker runner failed for tool '{tool.name}': {message}")

        stdout, stderr, artifacts = self._decode_response(proc.stdout, proc.stderr)
        return ToolExecutionResult(
            stdout=self._truncate_text(stdout),
            stderr=self._truncate_text(stderr),
            runner_used="docker",
            runtime_ms=runtime_ms,
            artifacts=artifacts,
        )

    def _build_command(self, env: Mapping[str, str]) -> List[str]:
        command = [
            self.docker_bin,
            "run",
            "--rm",
            "--interactive",
            "--workdir",
            "/workspace",
            "--network",
            self.network_default,
            "--cpus",
            str(self.cpu_limit),
            "--memory",
            f"{self.mem_limit_mb}m",
            "--pids-limit",
            str(self.pids_limit),
            "--mount",
            f"type=bind,src={self.workspace_root},dst=/workspace,rw",
        ]
        for key, value in env.items():
            command.extend(["--env", f"{key}={value}"])
        if self.read_only_root:
            command.extend(["--read-only", "--tmpfs", "/tmp:rw,size=64m"])
        for index, path in enumerate(self.read_paths):
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={path},dst=/sandbox/ro/{index},readonly",
                ]
            )
        for index, path in enumerate(self.write_paths):
            command.extend(
                [
                    "--mount",
                    f"type=bind,src={path},dst=/sandbox/rw/{index},rw",
                ]
            )
        command.append(self.image)
        command.extend(["python", "-m", self._ENTRYPOINT_MODULE])
        return command

    def _decode_response(
        self,
        stdout: str,
        stderr: str,
    ) -> tuple[str, str, List[str]]:
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

    def _normalize_read_paths(
        self,
        read_paths: Sequence[Path | str] | None,
    ) -> List[Path]:
        if not read_paths:
            return []
        normalized: List[Path] = []
        for item in read_paths:
            path = Path(item).expanduser().resolve(strict=False)
            if path == self.workspace_root:
                continue
            if path in normalized:
                continue
            normalized.append(path)
        return normalized

    def _normalize_write_paths(
        self,
        write_paths: Sequence[Path | str] | None,
    ) -> List[Path]:
        if not write_paths:
            return []
        normalized: List[Path] = []
        for item in write_paths:
            path = Path(item).expanduser().resolve(strict=False)
            if path == self.workspace_root:
                continue
            if path in normalized:
                continue
            normalized.append(path)
        return normalized

    def _translate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        def translate(value: Any) -> Any:
            if isinstance(value, str):
                return self._translate_path(value)
            if isinstance(value, list):
                return [translate(item) for item in value]
            if isinstance(value, dict):
                return {key: translate(item) for key, item in value.items()}
            return value

        return translate(dict(args))

    def _translate_path(self, value: str) -> str:
        text = str(value).strip()
        if not text:
            return value
        path = Path(text).expanduser()
        if not path.is_absolute():
            return value
        resolved = path.resolve(strict=False)
        candidate = self._map_to_container(resolved)
        return candidate if candidate is not None else value

    def _map_to_container(self, path: Path) -> str | None:
        from pathlib import PurePosixPath

        mappings: List[tuple[Path, str]] = [(self.workspace_root, "/workspace")]
        for index, root in enumerate(self.write_paths):
            mappings.append((root, f"/sandbox/rw/{index}"))
        for index, root in enumerate(self.read_paths):
            mappings.append((root, f"/sandbox/ro/{index}"))

        best_root: Path | None = None
        best_target: str | None = None
        best_relative: Path | None = None
        best_length = -1
        for root, target in mappings:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            length = len(root.parts)
            if length > best_length:
                best_root = root
                best_target = target
                best_length = length
                best_relative = relative
        if best_root is None or best_target is None or best_relative is None:
            return None
        return str(PurePosixPath(best_target) / PurePosixPath(best_relative.as_posix()))
