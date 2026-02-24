"""
Tool for summarizing evidence by source.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set, Tuple

from .base import BaseTool


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
                "description": "Evidence entries with source_path/snippet/score. May include node_id and node_type from file graph.",
                "items": {"type": "object"},
            },
            "max_snippets_per_source": {
                "type": "integer",
                "description": "Maximum snippets to include per source.",
            },
        },
        "required": ["evidence"],
    }

    def execute(self, args: Dict[str, Any]) -> str:
        raw_evidence = args.get("evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("evidence must be an array")

        max_snippets_per_source = max(1, int(args.get("max_snippets_per_source", 3)))
        counts: Dict[str, int] = {}
        snippets_by_source: Dict[str, List[Tuple[float, str]]] = {}
        node_ids_by_source: Dict[str, List[str]] = {}
        seen_node_ids_by_source: Dict[str, Set[str]] = {}

        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            source_path = str(item.get("source_path") or item.get("path") or "").strip()
            if not source_path:
                source_path = "unknown"

            score = self._coerce_float(item.get("score", item.get("relevance", 0.0)))
            snippet = str(
                item.get("snippet")
                or item.get("text")
                or item.get("excerpt")
                or ""
            ).strip()

            counts[source_path] = counts.get(source_path, 0) + 1
            if snippet:
                snippets_by_source.setdefault(source_path, []).append((score, snippet))

            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                seen = seen_node_ids_by_source.setdefault(source_path, set())
                if node_id not in seen:
                    seen.add(node_id)
                    node_ids_by_source.setdefault(source_path, []).append(node_id)

        output: Dict[str, Dict[str, Any]] = {}
        for source_path in sorted(counts.keys()):
            ranked_snippets = sorted(
                snippets_by_source.get(source_path, []),
                key=lambda value: value[0],
                reverse=True,
            )
            unique_snippets: List[str] = []
            seen_snippets: set[str] = set()
            for _, snippet in ranked_snippets:
                if snippet in seen_snippets:
                    continue
                seen_snippets.add(snippet)
                unique_snippets.append(snippet)
                if len(unique_snippets) >= max_snippets_per_source:
                    break

            entry: Dict[str, Any] = {
                "count": counts[source_path],
                "top_snippets": unique_snippets,
            }
            node_ids = node_ids_by_source.get(source_path)
            if node_ids:
                entry["node_ids"] = node_ids
            output[source_path] = entry

        return json.dumps(output, indent=2)

    def _coerce_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
