"""
Tool for file hashing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


class HashFileTool(BaseTool):
    """Compute SHA256 hash for a file."""

    name = "hash_file"
    description = "Compute SHA256 hash of a file."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)

        payload = {
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
        }
        return json.dumps(payload, indent=2)
