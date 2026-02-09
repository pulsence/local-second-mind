"""
Tool for intra-corpus similarity matching.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from lsm.vectordb.base import BaseVectorDBProvider, VectorDBGetResult

from .base import BaseTool


class SimilaritySearchTool(BaseTool):
    """Find similar chunk pairs for given chunk IDs or source paths."""

    name = "similarity_search"
    description = "Find similar chunk pairs for selected chunk IDs or paths."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "chunk_ids": {
                "type": "array",
                "description": "Chunk IDs to compare.",
                "items": {"type": "string"},
            },
            "paths": {
                "type": "array",
                "description": "Source paths whose chunks should be compared.",
                "items": {"type": "string"},
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of similar pairs to return.",
            },
            "threshold": {
                "type": "number",
                "description": "Minimum cosine similarity score to include.",
            },
        },
        "required": [],
    }

    def __init__(self, collection: BaseVectorDBProvider) -> None:
        self.collection = collection

    def execute(self, args: Dict[str, Any]) -> str:
        top_k = max(1, int(args.get("top_k", 20)))
        threshold = float(args.get("threshold", 0.75))

        raw_chunk_ids = args.get("chunk_ids")
        raw_paths = args.get("paths")
        chunk_ids = self._normalize_string_array(raw_chunk_ids)
        paths = self._normalize_string_array(raw_paths)

        if not chunk_ids and not paths:
            raise ValueError("At least one of chunk_ids or paths is required")

        records: Dict[str, Dict[str, Any]] = {}

        if chunk_ids:
            result = self.collection.get(
                ids=chunk_ids,
                include=["metadatas", "embeddings"],
            )
            self._merge_result_records(records, result)

        if paths:
            per_path_limit = max(50, top_k * 8)
            for source_path in paths:
                result = self.collection.get(
                    filters={"source_path": source_path},
                    limit=per_path_limit,
                    include=["metadatas", "embeddings"],
                )
                self._merge_result_records(records, result)

        entries = list(records.values())
        if len(entries) < 2:
            return "[]"

        pairs: List[Dict[str, Any]] = []
        for i in range(len(entries)):
            left = entries[i]
            for j in range(i + 1, len(entries)):
                right = entries[j]
                similarity = self._cosine_similarity(
                    left.get("embedding"),
                    right.get("embedding"),
                )
                if similarity is None or similarity < threshold:
                    continue
                pairs.append(
                    {
                        "id_a": left["id"],
                        "id_b": right["id"],
                        "source_path_a": left["source_path"],
                        "source_path_b": right["source_path"],
                        "similarity": round(similarity, 6),
                    }
                )

        pairs.sort(
            key=lambda item: (
                -float(item["similarity"]),
                str(item["id_a"]),
                str(item["id_b"]),
            )
        )
        return json.dumps(pairs[:top_k], indent=2)

    def _merge_result_records(
        self,
        records: Dict[str, Dict[str, Any]],
        result: VectorDBGetResult,
    ) -> None:
        metadatas = result.metadatas or []
        embeddings = result.embeddings or []
        for idx, chunk_id in enumerate(result.ids):
            embedding = embeddings[idx] if idx < len(embeddings) else None
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            if not embedding:
                continue
            records[str(chunk_id)] = {
                "id": str(chunk_id),
                "source_path": str(metadata.get("source_path", "")),
                "embedding": embedding,
            }

    def _normalize_string_array(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return list(dict.fromkeys(normalized))

    def _cosine_similarity(
        self,
        left: Optional[List[float]],
        right: Optional[List[float]],
    ) -> Optional[float]:
        if not left or not right:
            return None
        if len(left) != len(right):
            return None

        left_sq = sum(float(value) * float(value) for value in left)
        right_sq = sum(float(value) * float(value) for value in right)
        if left_sq <= 0 or right_sq <= 0:
            return None

        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        return float(dot / (math.sqrt(left_sq) * math.sqrt(right_sq)))
