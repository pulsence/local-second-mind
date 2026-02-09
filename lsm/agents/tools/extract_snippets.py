"""
Tool for path-scoped snippet extraction from local embeddings.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from lsm.query.retrieval import embed_text, retrieve_candidates
from lsm.vectordb.base import BaseVectorDBProvider

from .base import BaseTool


class ExtractSnippetsTool(BaseTool):
    """Extract top snippets for a query constrained to source paths."""

    name = "extract_snippets"
    description = "Extract relevant snippets from specific source paths."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Semantic query text."},
            "paths": {
                "type": "array",
                "description": "Source file paths to search within.",
                "items": {"type": "string"},
            },
            "max_snippets": {
                "type": "integer",
                "description": "Maximum snippets to return.",
            },
            "max_chars_per_snippet": {
                "type": "integer",
                "description": "Maximum characters per returned snippet.",
            },
        },
        "required": ["query", "paths"],
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

        raw_paths = args.get("paths")
        if not isinstance(raw_paths, list):
            raise ValueError("paths must be an array of strings")
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        if not paths:
            raise ValueError("paths must contain at least one path")

        max_snippets = max(1, int(args.get("max_snippets", 8)))
        max_chars = int(args.get("max_chars_per_snippet", 400))
        if max_chars <= 0:
            max_chars = 400

        unique_paths = list(dict.fromkeys(paths))
        per_path_k = max(1, max_snippets)
        query_embedding = embed_text(self.embedder, query, batch_size=self.batch_size)

        collected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for source_path in unique_paths:
            candidates = retrieve_candidates(
                self.collection,
                query_embedding,
                per_path_k,
                where_filter={"source_path": source_path},
            )
            for candidate in candidates:
                if candidate.cid in seen_ids:
                    continue
                seen_ids.add(candidate.cid)
                snippet = candidate.text
                if max_chars > 0:
                    snippet = snippet[:max_chars]
                collected.append(
                    {
                        "source_path": str(candidate.meta.get("source_path", source_path)),
                        "snippet": snippet,
                        "score": round(float(candidate.relevance), 6),
                    }
                )

        collected.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return json.dumps(collected[:max_snippets], indent=2)
