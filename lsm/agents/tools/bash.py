"""
Bash command execution tool.
"""

from __future__ import annotations

import subprocess
from typing import Any, Dict

from .base import BaseTool


class BashTool(BaseTool):
    """Execute a bash command string."""

    name = "bash"
    description = "Execute a bash command string."
    risk_level = "exec"
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute."},
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

        proc = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout_value,
            check=False,
        )
        stdout = str(proc.stdout or "")
        stderr = str(proc.stderr or "")
        if proc.returncode != 0:
            message = stderr.strip() or stdout.strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"Bash command failed: {message}")
        if stdout.strip():
            return stdout
        return stderr
