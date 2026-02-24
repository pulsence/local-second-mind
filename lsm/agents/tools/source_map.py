"""
Tool for summarizing evidence by source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from .base import BaseTool
from lsm.utils.file_graph import build_graph_outline, get_file_graph


class SourceMapTool(BaseTool):
    """Aggregate evidence entries into a source-centric map."""

    name = "source_map"
    description = "Build a source map from evidence snippets."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "array",
                "description": "Evidence entries with source_path and optional node_id metadata.",
                "items": {"type": "object"},
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum outline depth to include per source.",
            },
            "node_type": {
                "type": "string",
                "description": "Optional graph node type filter for outlines.",
            },
        },
        "required": ["evidence"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        raw_evidence = args.get("evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("evidence must be an array")

        max_depth = max(1, int(args.get("max_depth", 2)))
        node_type = str(args.get("node_type") or "").strip()
        counts: Dict[str, int] = {}
        node_ids_by_source: Dict[str, List[str]] = {}
        seen_node_ids_by_source: Dict[str, Set[str]] = {}

        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            source_path = str(item.get("source_path") or item.get("path") or "").strip()
            if not source_path:
                source_path = "unknown"

            counts[source_path] = counts.get(source_path, 0) + 1

            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                seen = seen_node_ids_by_source.setdefault(source_path, set())
                if node_id not in seen:
                    seen.add(node_id)
                    node_ids_by_source.setdefault(source_path, []).append(node_id)

        output: Dict[str, Dict[str, Any]] = {}
        for source_path in sorted(counts.keys()):
            entry: Dict[str, Any] = {"count": counts[source_path], "outline": []}
            outline_error = None
            if source_path != "unknown":
                path = Path(source_path)
                if path.exists() and path.is_file():
                    graph = get_file_graph(path)
                    entry["outline"] = build_graph_outline(
                        graph,
                        max_depth=max_depth,
                        node_types=[node_type] if node_type else None,
                    )
                else:
                    outline_error = "File not found"
            node_ids = node_ids_by_source.get(source_path)
            if node_ids:
                entry["node_ids"] = node_ids
            if outline_error:
                entry["outline_error"] = outline_error
            output[source_path] = entry

        return json.dumps(output, indent=2)
