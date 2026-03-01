"""
Tool for running the full query pipeline (build_sources → synthesize → execute).

Replaces query_knowledge_base with pipeline-backed execution.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.config.models.agents import SandboxConfig
    from lsm.query.pipeline import RetrievalPipeline


class QueryAndSynthesizeTool(BaseTool):
    """Run the full retrieval pipeline and return a synthesized answer with sources."""

    name = "query_and_synthesize"
    description = (
        "Query the knowledge base and produce a grounded answer with citations. "
        "Runs the full pipeline: retrieval, context assembly, and LLM synthesis."
    )
    risk_level = "network"
    needs_network = True
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
                "description": "Explicit synthesis prompt (overrides mode default).",
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

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        sandbox_config: Optional["SandboxConfig"] = None,
    ) -> None:
        self.pipeline = pipeline
        self.sandbox_config = sandbox_config

    def execute(self, args: Dict[str, Any]) -> str:
        from lsm.query.pipeline_types import FilterSet, QueryRequest
        from .mode_validation import validate_agent_mode

        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        mode = args.get("mode")
        error = validate_agent_mode(mode, self.pipeline.config, self.sandbox_config)
        if error:
            raise ValueError(error)

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

        response = self.pipeline.run(request)

        output = response.to_dict()
        # Include candidate summaries for agent consumption
        candidates_out = []
        for c in response.candidates[:10]:
            entry: Dict[str, Any] = {"id": c.cid, "text": c.text[:500]}
            if hasattr(c, "relevance"):
                entry["relevance"] = c.relevance
            if c.meta:
                entry["source_path"] = c.meta.get("source_path", "")
            candidates_out.append(entry)
        output["candidates"] = candidates_out

        return json.dumps(output, indent=2)
