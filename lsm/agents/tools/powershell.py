"""
PowerShell command execution tool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict

from .base import BaseTool


class PowerShellTool(BaseTool):
    """Execute a PowerShell command string."""

    name = "powershell"
    description = "Execute a PowerShell command string."
    risk_level = "exec"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "PowerShell command to execute."},
            "timeout_s": {"type": "number", "description": "Optional timeout override in seconds."},
        },
        "required": ["command"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command is required")
        timeout = args.get("timeout_s")
        timeout_value = float(timeout) if timeout is not None else None

        shell = self._resolve_shell()
        proc = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_value,
            check=False,
        )
        stdout = str(proc.stdout or "")
        stderr = str(proc.stderr or "")
        if proc.returncode != 0:
            message = stderr.strip() or stdout.strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"PowerShell command failed: {message}")
        if stdout.strip():
            return stdout
        return stderr

    def _resolve_shell(self) -> str:
        if os.name == "nt":
            return "powershell"
        for candidate in ("pwsh", "powershell"):
            if shutil.which(candidate):
                return candidate
        raise RuntimeError("PowerShell executable not found (pwsh or powershell)")
