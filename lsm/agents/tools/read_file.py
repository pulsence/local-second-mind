"""
Tool for reading a file from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseTool
from lsm.utils.file_graph import (
    build_graph_outline,
    compute_line_hashes,
    get_file_graph,
    get_graph_text,
)


class ReadFileTool(BaseTool):
    """Read UTF-8 text from a file."""

    name = "read_file"
    description = "Read text content from a file path."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "section": {
                "type": "string",
                "description": "Optional section name or graph node id to read.",
            },
            "node_type": {
                "type": "string",
                "description": "Optional node type filter (function/class/heading).",
            },
            "max_depth": {
                "type": "integer",
                "description": "Optional outline depth to include in structured output.",
            },
            "include_hashes": {
                "type": "boolean",
                "description": "Include per-line hashes when true.",
            },
        },
        "required": ["path"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        path = Path(str(args.get("path", "")).strip())
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        section = str(args.get("section") or "").strip()
        node_type = str(args.get("node_type") or "").strip()
        include_hashes = bool(args.get("include_hashes", False))
        max_depth = args.get("max_depth")
        max_depth = int(max_depth) if max_depth is not None else None

        return_structured = bool(section or include_hashes or max_depth is not None)

        text = get_graph_text(path)
        lines = text.split("\n") if text else [""]

        selected_node: Optional[object] = None
        if section:
            graph = get_file_graph(path)
            selected_node = next((item for item in graph.nodes if item.id == section), None)
            if selected_node is None:
                default_types = {"heading", "function", "class"}
                for node in graph.nodes:
                    if node_type:
                        if node.node_type != node_type:
                            continue
                    elif node.node_type not in default_types:
                        continue
                    if section.lower() in node.name.lower():
                        selected_node = node
                        break

            if selected_node is None:
                raise ValueError("Graph node not found for requested section")

        if not return_structured:
            if selected_node is None:
                return text
            start = max(0, selected_node.start_line - 1)
            end = min(selected_node.end_line, len(lines))
            return "\n".join(lines[start:end])

        content = text
        start_idx = 0
        end_idx = len(lines)
        if selected_node is not None:
            start_idx = max(0, selected_node.start_line - 1)
            end_idx = min(selected_node.end_line, len(lines))
            content = "\n".join(lines[start_idx:end_idx])

        graph = get_file_graph(path)
        outline = build_graph_outline(graph, max_depth=max_depth or 2)

        payload: Dict[str, Any] = {
            "path": str(path.resolve()),
            "content": content,
            "outline": outline,
        }

        if selected_node is not None:
            payload["section"] = {"node": selected_node.to_dict()}

        if include_hashes:
            hashes = compute_line_hashes(lines)
            if selected_node is not None:
                payload["line_hashes"] = [
                    {"line": idx + 1, "hash": hashes[idx]}
                    for idx in range(start_idx, end_idx)
                ]
                if hashes:
                    payload["start_hash"] = hashes[start_idx]
                    payload["end_hash"] = hashes[end_idx - 1] if end_idx > 0 else ""
            else:
                payload["line_hashes"] = [
                    {"line": idx + 1, "hash": value} for idx, value in enumerate(hashes)
                ]

        return json.dumps(payload, indent=2)
