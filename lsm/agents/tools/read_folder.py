"""
Tool for listing folder contents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool


class ReadFolderTool(BaseTool):
    """List files/directories in a folder."""

    name = "read_folder"
    description = "List files and directories within a folder."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the folder."},
            "recursive": {
                "type": "boolean",
                "description": "If true, include descendants recursively.",
            },
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        recursive = bool(args.get("recursive", False))
        if not path.exists():
            raise FileNotFoundError(f"Folder not found: {path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a folder: {path}")

        entries: List[Dict[str, Any]] = []
        iterator = path.rglob("*") if recursive else path.iterdir()
        for item in iterator:
            entries.append(
                {
                    "name": item.name,
                    "path": str(item.resolve()),
                    "is_dir": item.is_dir(),
                }
            )
        return json.dumps(entries, indent=2)
