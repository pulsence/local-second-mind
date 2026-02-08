"""
Tool for local vector similarity search.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.query.retrieval import embed_text, retrieve_candidates
from lsm.vectordb.base import BaseVectorDBProvider

from .base import BaseTool


class QueryEmbeddingsTool(BaseTool):
    """Query local embeddings through the configured vector DB."""

    name = "query_embeddings"
    description = "Run local semantic search over the vector database."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic query text."},
            "top_k": {"type": "integer", "description": "Number of results to return."},
            "filters": {"type": "object", "description": "Optional metadata filters."},
            "max_chars": {"type": "integer", "description": "Max chars of text per hit."},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        collection: BaseVectorDBProvider,
        embedder: Any,
        batch_size: int = 32,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.batch_size = batch_size

    def execute(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        top_k = int(args.get("top_k", 5))
        filters = args.get("filters")
        max_chars = int(args.get("max_chars", 500))
        query_embedding = embed_text(self.embedder, query, batch_size=self.batch_size)
        candidates = retrieve_candidates(
            self.collection,
            query_embedding,
            top_k,
            where_filter=filters if isinstance(filters, dict) else None,
        )
        payload = []
        for candidate in candidates:
            payload.append(
                {
                    "id": candidate.cid,
                    "distance": candidate.distance,
                    "relevance": candidate.relevance,
                    "text": candidate.text[:max_chars] if max_chars > 0 else candidate.text,
                    "metadata": candidate.meta,
                }
            )
        return json.dumps(payload, indent=2)

