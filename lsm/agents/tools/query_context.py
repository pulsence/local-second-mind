"""
Tool for querying the knowledge base retrieval pipeline (Stage 1: build_sources).

Returns a serialized ContextPackage that agents can inspect before synthesis.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.query.pipeline import RetrievalPipeline


class QueryContextTool(BaseTool):
    """Retrieve context from the knowledge base without LLM synthesis."""

    name = "query_context"
    description = (
        "Search the knowledge base and return raw context (candidates and sources) "
        "without LLM synthesis. Use this when you want to inspect retrieved chunks "
        "before deciding how to synthesize an answer."
    )
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query text to search the knowledge base.",
            },
            "mode": {
                "type": "string",
                "description": "Query mode name (e.g. 'grounded', 'insight', 'hybrid').",
            },
            "k": {
                "type": "integer",
                "description": "Number of top candidates to retrieve.",
            },
            "filters": {
                "type": "object",
                "description": "Metadata filters: path_contains, ext_allow, ext_deny.",
                "properties": {
                    "path_contains": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ext_allow": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "ext_deny": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "starting_prompt": {
                "type": "string",
                "description": "Explicit synthesis prompt to store in the context package.",
            },
            "conversation_id": {
                "type": "string",
                "description": "Conversation session identifier for multi-turn chaining.",
            },
            "prior_response_id": {
                "type": "string",
                "description": "Response ID from the previous turn for cache chaining.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, pipeline: "RetrievalPipeline") -> None:
        self.pipeline = pipeline

    def execute(self, args: Dict[str, Any]) -> str:
        from lsm.query.pipeline_types import FilterSet, QueryRequest

        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        filters: Optional[FilterSet] = None
        raw_filters = args.get("filters")
        if isinstance(raw_filters, dict):
            filters = FilterSet(
                path_contains=raw_filters.get("path_contains"),
                ext_allow=raw_filters.get("ext_allow"),
                ext_deny=raw_filters.get("ext_deny"),
            )

        request = QueryRequest(
            question=query,
            mode=args.get("mode"),
            filters=filters,
            k=args.get("k"),
            starting_prompt=args.get("starting_prompt"),
            conversation_id=args.get("conversation_id"),
            prior_response_id=args.get("prior_response_id"),
        )

        package = self.pipeline.build_sources(request)

        candidates_out: List[Dict[str, Any]] = []
        for c in package.candidates:
            entry: Dict[str, Any] = {"id": c.cid, "text": c.text[:500]}
            if hasattr(c, "relevance"):
                entry["relevance"] = c.relevance
            if hasattr(c, "distance"):
                entry["distance"] = c.distance
            if c.meta:
                entry["source_path"] = c.meta.get("source_path", "")
                entry["heading"] = c.meta.get("heading_text", "")
            candidates_out.append(entry)

        output: Dict[str, Any] = {
            "candidates": candidates_out,
            "candidate_count": len(candidates_out),
            "retrieval_trace": package.retrieval_trace.to_dict(),
            "relevance": package.relevance,
            "local_enabled": package.local_enabled,
        }
        if package.remote_sources:
            output["remote_sources"] = [rs.to_dict() for rs in package.remote_sources]
        if package.costs:
            output["costs"] = [c.to_dict() for c in package.costs]
        if package.request.conversation_id:
            output["conversation_id"] = package.request.conversation_id
        if package.prior_response_id:
            output["prior_response_id"] = package.prior_response_id

        return json.dumps(output, indent=2)
