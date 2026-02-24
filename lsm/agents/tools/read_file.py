"""
Tool for reading a file from disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import BaseTool
from lsm.utils.file_graph import get_file_graph, get_graph_text


class ReadFileTool(BaseTool):
    """Read UTF-8 text from a file."""

    name = "read_file"
    description = "Read text content from a file path."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "node_id": {
                "type": "string",
                "description": "Optional graph node id to read a specific section.",
            },
            "node_type": {
                "type": "string",
                "description": "Optional node type to match (used with name).",
            },
            "name": {
                "type": "string",
                "description": "Optional node name to match (used with node_type).",
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

        node_id = args.get("node_id")
        node_type = args.get("node_type")
        name = args.get("name")

        if node_id or node_type:
            graph = get_file_graph(path)
            node = None
            if node_id:
                node = next((item for item in graph.nodes if item.id == node_id), None)
            elif node_type and name:
                node = next(
                    (item for item in graph.nodes if item.node_type == node_type and item.name == name),
                    None,
                )

            if node is None:
                raise ValueError("Graph node not found for requested selector")

            text = get_graph_text(path)
            lines = text.split("\n") if text else []
            start = max(0, node.start_line - 1)
            end = min(node.end_line, len(lines))
            return "\n".join(lines[start:end])

        return path.read_text(encoding="utf-8")
