"""
Tool for graph-aware file discovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool
from lsm.utils.file_graph import build_graph_outline, get_file_graph, get_graph_text


class FindFileTool(BaseTool):
    """Find files by name or content pattern using file graphs."""

    name = "find_file"
    description = "Search for files by name/content patterns and return structural outlines."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root directory to search.",
            },
            "name_pattern": {
                "type": "string",
                "description": "Filename pattern or regex to match.",
            },
            "content_pattern": {
                "type": "string",
                "description": "Content pattern or regex to match within files.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether pattern matching is case sensitive.",
            },
            "use_regex": {
                "type": "boolean",
                "description": "Treat patterns as regular expressions when true.",
            },
            "language": {
                "type": "string",
                "description": "Optional language filter for graph nodes.",
            },
            "node_type": {
                "type": "string",
                "description": "Optional graph node type filter (function/class/heading).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum outline depth to return.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
            },
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        root = Path(str(args.get("path", "")).strip())
        if not root.exists():
            raise FileNotFoundError(f"Folder not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Path is not a folder: {root}")

        name_pattern = str(args.get("name_pattern") or "").strip()
        content_pattern = str(args.get("content_pattern") or "").strip()
        if not name_pattern and not content_pattern:
            raise ValueError("name_pattern or content_pattern is required")

        case_sensitive = bool(args.get("case_sensitive", False))
        use_regex = bool(args.get("use_regex", False))
        max_results = int(args.get("max_results", 25))
        max_depth = args.get("max_depth")
        max_depth = int(max_depth) if max_depth is not None else 2
        language = str(args.get("language") or "").strip().lower()
        node_type = str(args.get("node_type") or "").strip()

        def compile_pattern(pattern: str) -> Optional[re.Pattern[str]]:
            if not pattern or not use_regex:
                return None
            flags = 0 if case_sensitive else re.IGNORECASE
            return re.compile(pattern, flags=flags)

        name_re = compile_pattern(name_pattern)
        content_re = compile_pattern(content_pattern)

        def match_value(value: str, pattern: str, regex: Optional[re.Pattern[str]]) -> bool:
            if not pattern:
                return False
            if regex is not None:
                return bool(regex.search(value))
            if not case_sensitive:
                return pattern.lower() in value.lower()
            return pattern in value

        def matches_language(path: Path, graph) -> bool:
            if not language:
                return True
            for node in graph.nodes:
                node_lang = str(node.metadata.get("language", "")).lower()
                if node_lang == language:
                    return True
            ext = path.suffix.lower()
            language_ext_map = {
                "python": {".py", ".pyw"},
                "javascript": {".js", ".jsx"},
                "typescript": {".ts", ".tsx"},
                "markdown": {".md"},
                "text": {".txt", ".rst"},
                "html": {".html", ".htm"},
            }
            return ext in language_ext_map.get(language, set())

        entries: List[Dict[str, Any]] = []
        for item in root.rglob("*"):
            if not item.is_file():
                continue

            name_match = match_value(item.name, name_pattern, name_re) if name_pattern else False
            content_match = False
            if content_pattern:
                try:
                    content = get_graph_text(item)
                except Exception:
                    content = ""
                content_match = match_value(content, content_pattern, content_re)

            if name_pattern and content_pattern:
                if not (name_match or content_match):
                    continue
            elif name_pattern and not name_match:
                continue
            elif content_pattern and not content_match:
                continue

            graph = get_file_graph(item)
            if node_type and not any(node.node_type == node_type for node in graph.nodes):
                continue
            if not matches_language(item, graph):
                continue

            outline = build_graph_outline(
                graph,
                max_depth=max_depth,
                node_types=[node_type] if node_type else None,
            )
            entries.append(
                {
                    "path": str(item.resolve()),
                    "name": item.name,
                    "matches": {"name": name_match, "content": content_match},
                    "outline": outline,
                }
            )
            if max_results and len(entries) >= max_results:
                break

        return json.dumps(entries, indent=2)
