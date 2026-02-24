"""
Tool for line-hash based file edits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool
from lsm.utils.file_graph import HTML_EXTENSIONS, compute_line_hashes, get_file_graph


class EditFileTool(BaseTool):
    """Apply deterministic edits to a file using line hashes."""

    name = "edit_file"
    description = "Edit a file by replacing a line range identified by hashes."
    risk_level = "writes_workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "start_hash": {
                "type": "string",
                "description": "Line hash for the start of the replacement range.",
            },
            "end_hash": {
                "type": "string",
                "description": "Line hash for the end of the replacement range.",
            },
            "new_content": {
                "type": "string",
                "description": "Replacement text to insert for the range.",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional start line number to disambiguate hash collisions.",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional end line number to disambiguate hash collisions.",
            },
        },
        "required": ["path", "start_hash", "end_hash", "new_content"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix in HTML_EXTENSIONS or suffix in {".pdf", ".docx"}:
            raise ValueError("edit_file does not support HTML/PDF/DOCX formats")

        start_hash = str(args.get("start_hash", "")).strip()
        end_hash = str(args.get("end_hash", "")).strip()
        new_content = str(args.get("new_content", ""))
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        if not start_hash or not end_hash:
            raise ValueError("start_hash and end_hash are required")

        text = path.read_text(encoding="utf-8")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n") if text else [""]
        hashes = compute_line_hashes(lines)

        start_matches = [idx for idx, value in enumerate(hashes) if value == start_hash]
        end_matches = [idx for idx, value in enumerate(hashes) if value == end_hash]

        def build_context(indices: List[int], window: int = 2) -> List[Dict[str, Any]]:
            contexts: List[Dict[str, Any]] = []
            for idx in indices[:3]:
                start = max(0, idx - window)
                end = min(len(lines) - 1, idx + window)
                window_lines = [
                    {
                        "line": pos + 1,
                        "hash": hashes[pos],
                        "text": lines[pos],
                    }
                    for pos in range(start, end + 1)
                ]
                contexts.append({"line": idx + 1, "window": window_lines})
            return contexts

        def error_payload(
            error: str,
            message: str,
            *,
            actual_hashes: Dict[str, Any] | None = None,
        ) -> str:
            details: Dict[str, Any] = {
                "start_hash": start_hash,
                "end_hash": end_hash,
                "start_matches": [idx + 1 for idx in start_matches],
                "end_matches": [idx + 1 for idx in end_matches],
                "context": {
                    "start": build_context(start_matches),
                    "end": build_context(end_matches),
                },
                "suggestions": {
                    "use_start_line": [idx + 1 for idx in start_matches[:5]],
                    "use_end_line": [idx + 1 for idx in end_matches[:5]],
                },
            }
            if actual_hashes:
                details["actual_hashes"] = actual_hashes
            else:
                details["actual_hashes"] = {}
            return json.dumps(
                {
                    "status": "error",
                    "error": error,
                    "message": message,
                    "details": details,
                },
                indent=2,
            )

        def resolve_index(
            label: str,
            target_hash: str,
            line_number: Any,
            matches: List[int],
        ) -> int | None:
            if not matches:
                return None
            if line_number is not None:
                try:
                    line_no = int(line_number)
                except (TypeError, ValueError):
                    return None
                idx = line_no - 1
                if idx < 0 or idx >= len(lines):
                    return None
                if hashes[idx] != target_hash:
                    return None
                return idx
            if len(matches) > 1:
                return None
            return matches[0]

        start_idx = resolve_index("start", start_hash, start_line, start_matches)
        end_idx = resolve_index("end", end_hash, end_line, end_matches)

        if start_idx is None or end_idx is None:
            actual_hashes: Dict[str, Any] = {}
            if start_line is not None:
                try:
                    idx = int(start_line) - 1
                    if 0 <= idx < len(hashes):
                        actual_hashes["start_line"] = {
                            "line": idx + 1,
                            "hash": hashes[idx],
                            "text": lines[idx],
                        }
                except (TypeError, ValueError):
                    pass
            if end_line is not None:
                try:
                    idx = int(end_line) - 1
                    if 0 <= idx < len(hashes):
                        actual_hashes["end_line"] = {
                            "line": idx + 1,
                            "hash": hashes[idx],
                            "text": lines[idx],
                        }
                except (TypeError, ValueError):
                    pass

            if (len(start_matches) > 1 and start_line is None) or (
                len(end_matches) > 1 and end_line is None
            ):
                return error_payload(
                    "hash_collision",
                    "Hash collision detected; provide start_line/end_line to disambiguate.",
                    actual_hashes=actual_hashes,
                )
            return error_payload(
                "hash_mismatch",
                "Provided line hashes did not match the file contents.",
                actual_hashes=actual_hashes,
            )

        if start_idx > end_idx:
            return error_payload(
                "range_invalid",
                "start_hash occurs after end_hash; provide the correct range.",
            )

        if new_content == "":
            new_lines: List[str] = []
        else:
            new_lines = new_content.split("\n")

        updated_lines = lines[:start_idx] + new_lines + lines[end_idx + 1 :]
        updated_text = "\n".join(updated_lines)
        path.write_text(updated_text, encoding="utf-8")

        graph = get_file_graph(path)

        return json.dumps(
            {
                "status": "ok",
                "path": str(path.resolve()),
                "start_line": start_idx + 1,
                "end_line": end_idx + 1,
                "replaced_lines": end_idx - start_idx + 1,
                "new_lines": len(new_lines),
                "graph": graph.to_dict(),
            },
            indent=2,
        )
