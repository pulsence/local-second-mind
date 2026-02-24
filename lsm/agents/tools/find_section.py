"""
Tool for graph-aware section discovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool
from lsm.utils.file_graph import compute_line_hashes, get_file_graph, get_graph_text


class FindSectionTool(BaseTool):
    """Find sections within files by heading/function/class name."""

    name = "find_section"
    description = "Locate sections within files using structural graphs."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to search.",
            },
            "paths": {
                "type": "array",
                "description": "Optional list of file paths to search.",
                "items": {"type": "string"},
            },
            "section": {
                "type": "string",
                "description": "Section name or node id to match.",
            },
            "node_type": {
                "type": "string",
                "description": "Optional graph node type filter (function/class/heading).",
            },
            "use_regex": {
                "type": "boolean",
                "description": "Treat section name as a regex when true.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether section matching is case sensitive.",
            },
            "language": {
                "type": "string",
                "description": "Optional language filter for graph nodes.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of sections to return.",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum outline depth to return.",
            },
        },
        "required": ["section"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        section = str(args.get("section") or "").strip()
        if not section:
            raise ValueError("section is required")

        raw_paths = args.get("paths")
        path_arg = str(args.get("path") or "").strip()
        paths: List[Path] = []
        if isinstance(raw_paths, list):
            for item in raw_paths:
                if str(item).strip():
                    paths.append(Path(str(item).strip()))
        if path_arg:
            paths.append(Path(path_arg))

        if not paths:
            raise ValueError("path or paths must be provided")

        node_type = str(args.get("node_type") or "").strip()
        use_regex = bool(args.get("use_regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        language = str(args.get("language") or "").strip().lower()
        max_results = int(args.get("max_results", 10))
        max_depth = args.get("max_depth")
        max_depth = int(max_depth) if max_depth is not None else None

        flags = 0 if case_sensitive else re.IGNORECASE
        section_re: Optional[re.Pattern[str]] = re.compile(section, flags=flags) if use_regex else None

        def match_name(value: str) -> bool:
            if section_re is not None:
                return bool(section_re.search(value))
            if not case_sensitive:
                return section.lower() in value.lower()
            return section in value

        def matches_language(graph) -> bool:
            if not language:
                return True
            for node in graph.nodes:
                node_lang = str(node.metadata.get("language", "")).lower()
                if node_lang == language:
                    return True
            return False

        entries: List[Dict[str, Any]] = []
        default_types = {"heading", "function", "class"}

        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")

            graph = get_file_graph(path)
            if not matches_language(graph):
                continue

            text = get_graph_text(path)
            lines = text.split("\n") if text else [""]
            line_hashes = compute_line_hashes(lines)

            for node in graph.nodes:
                if max_depth is not None and node.depth > max_depth:
                    continue
                if node_type:
                    if node.node_type != node_type:
                        continue
                else:
                    if node.node_type not in default_types:
                        continue

                if section == node.id or match_name(node.name):
                    start_idx = max(node.start_line - 1, 0)
                    end_idx = min(node.end_line, len(lines))
                    content = "\n".join(lines[start_idx:end_idx])
                    section_hashes = [
                        {"line": idx + 1, "hash": line_hashes[idx]}
                        for idx in range(start_idx, end_idx)
                    ]
                    start_hash = line_hashes[start_idx] if line_hashes else ""
                    end_hash = line_hashes[end_idx - 1] if end_idx > 0 else ""

                    entries.append(
                        {
                            "path": str(path.resolve()),
                            "node": node.to_dict(),
                            "content": content,
                            "line_hashes": section_hashes,
                            "start_hash": start_hash,
                            "end_hash": end_hash,
                        }
                    )
                    if max_results and len(entries) >= max_results:
                        return json.dumps(entries, indent=2)

        return json.dumps(entries, indent=2)
