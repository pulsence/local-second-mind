"""
Tool for writing text to a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class WriteFileTool(BaseTool):
    """Write UTF-8 text content to a file."""

    name = "write_file"
    description = "Write text content to a file path."
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write."},
            "content": {"type": "string", "description": "Text content to write."},
            "append": {
                "type": "boolean",
                "description": "Append instead of overwrite when true.",
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        content = str(args.get("content", ""))
        append = bool(args.get("append", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return f"Wrote {len(content)} chars to {path.resolve()}"

