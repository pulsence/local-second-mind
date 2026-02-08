"""
Tool for creating directories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class CreateFolderTool(BaseTool):
    """Create a folder on disk."""

    name = "create_folder"
    description = "Create a directory path."
    requires_permission = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to create."},
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        path.mkdir(parents=True, exist_ok=True)
        return f"Created folder: {path.resolve()}"

