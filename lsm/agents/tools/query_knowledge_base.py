"""
Tool for querying the knowledge base using the full query pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from lsm.query.api import query_sync
from lsm.query.session import SessionState

from .base import BaseTool


class QueryKnowledgeBaseTool(BaseTool):
    """Query the knowledge base using the full query pipeline including reranking and LLM synthesis."""

    name = "query_knowledge_base"
    description = (
        "Query the knowledge base using the full pipeline including reranking and LLM synthesis. "
        "Returns a grounded answer with sources."
    )
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query text to search the knowledge base.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top candidates to retrieve.",
            },
            "filters": {
                "type": "object",
                "description": "Optional metadata filters for the query.",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        config: Any,
        embedder: Any,
        collection: Any,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.collection = collection

    def execute(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        top_k = args.get("top_k")
        filters = args.get("filters")
        max_chars = int(args.get("max_chars", 500))

        state = SessionState()

        result = query_sync(
            question=query,
            config=self.config,
            state=state,
            embedder=self.embedder,
            collection=self.collection,
        )

        top_candidates = []
        for candidate in result.candidates[: (top_k if top_k is not None else 5)]:
            top_candidates.append({
                "id": candidate.cid,
                "score": candidate.relevance,
                "text": candidate.text[:max_chars] if max_chars > 0 else candidate.text,
            })

        output = {
            "answer": result.answer,
            "sources_display": result.sources_display,
            "candidates": top_candidates,
        }

        return json.dumps(output, indent=2)
