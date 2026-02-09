"""
Tool for reading a file from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class ReadFileTool(BaseTool):
    """Read UTF-8 text from a file."""

    name = "read_file"
    description = "Read text content from a file path."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path.read_text(encoding="utf-8")
