"""
Tool for reading filesystem metadata for paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool


class FileMetadataTool(BaseTool):
    """Return metadata for one or more file system paths."""

    name = "file_metadata"
    description = "Return size, mtime, and extension metadata for paths."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "description": "Paths to inspect.",
                "items": {"type": "string"},
            }
        },
        "required": ["paths"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        raw_paths = args.get("paths")
        if not isinstance(raw_paths, list):
            raise ValueError("paths must be an array of strings")
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        if not paths:
            raise ValueError("paths must contain at least one path")

        payload: List[Dict[str, Any]] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")

            stats = path.stat()
            payload.append(
                {
                    "path": str(path.resolve()),
                    "size_bytes": int(stats.st_size),
                    "mtime_iso": datetime.fromtimestamp(
                        stats.st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                    "ext": path.suffix.lower(),
                }
            )

        return json.dumps(payload, indent=2)
