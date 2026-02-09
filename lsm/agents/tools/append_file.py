"""
Tool for appending text to a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class AppendFileTool(BaseTool):
    """Append UTF-8 text content to an existing or new file."""

    name = "append_file"
    description = "Append text content to a file path."
    requires_permission = True
    risk_level = "writes_workspace"
    needs_network = False
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to append to."},
            "content": {"type": "string", "description": "Text content to append."},
        },
        "required": ["path", "content"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return f"Appended {len(content)} chars to {path.resolve()}"
