"""
LLM reranking stage prompt assets and orchestration.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from lsm.logging import get_logger
from lsm.providers.base import BaseLLMProvider
from lsm.providers.helpers import parse_json_payload

logger = get_logger(__name__)


RERANK_INSTRUCTIONS = """You are a retrieval reranker.
Goal: rank the candidate passages by how useful they are for answering the user's question.
Guidance:
- Prefer passages that directly address the question.
- Prefer specificity, definitions, arguments, or evidence over vague mentions.
- If multiple passages are similar, rank the most comprehensive/precise first.
- Do NOT hallucinate facts; you are only ranking.

Output requirements:
- Return STRICT JSON only, no markdown, no extra text.
- Schema: {{"ranking":[{{"index":int,"reason":string}}...]}}
- Include at most {k} items.
"""


RERANK_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranking"],
    "additionalProperties": False,
}


def prepare_candidates_for_rerank(
    candidates: List[Dict[str, Any]],
    max_text_length: int = 1200,
) -> List[Dict[str, Any]]:
    """Prepare rerank payload with truncation and source metadata."""
    items: List[Dict[str, Any]] = []
    truncate_at = max_text_length - 50

    for i, cand in enumerate(candidates):
        text = cand.get("text", "")
        metadata = cand.get("metadata", {})

        if len(text) > max_text_length:
            text = text[:truncate_at] + "\n...[truncated]..."

        items.append(
            {
                "index": i,
                "source_path": metadata.get("source_path", "unknown"),
                "source_name": metadata.get("source_name"),
                "chunk_index": metadata.get("chunk_index"),
                "ext": metadata.get("ext"),
                "distance": cand.get("distance"),
                "text": text,
            }
        )

    return items


def parse_ranking_response(
    ranking: Any,
    candidates: List[Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Convert model ranking payload into ordered candidates."""
    if not isinstance(ranking, list):
        return candidates[:k]

    chosen: List[Dict[str, Any]] = []
    seen: Set[int] = set()

    for item in ranking:
        if not isinstance(item, dict) or "index" not in item:
            continue
        try:
            idx = int(item["index"])
        except (TypeError, ValueError):
            continue

        if 0 <= idx < len(candidates) and idx not in seen:
            chosen.append(candidates[idx])
            seen.add(idx)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        for i, candidate in enumerate(candidates):
            if i not in seen:
                chosen.append(candidate)
            if len(chosen) >= k:
                break

    return chosen


def llm_rerank(
    candidates: List[Dict[str, Any]],
    query: str,
    provider: BaseLLMProvider,
    *,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate dictionaries with an LLM.

    Falls back to local candidate order when provider output is invalid.
    """
    if not candidates:
        return []

    top_k = max(1, min(k, len(candidates)))
    payload = {
        "question": query,
        "top_n": top_k,
        "candidates": prepare_candidates_for_rerank(candidates),
    }
    instructions = RERANK_INSTRUCTIONS.format(k=top_k)

    try:
        raw = provider.send_message(
            input=json.dumps(payload),
            instruction=instructions,
            temperature=0.2,
            max_tokens=800,
            json_schema=RERANK_JSON_SCHEMA,
            json_schema_name="rerank_response",
            reasoning_effort="low",
        )
        data = parse_json_payload(raw)
        ranking = data.get("ranking", []) if isinstance(data, dict) else None
        if not isinstance(ranking, list):
            logger.warning(
                "LLM rerank returned invalid structure (%s/%s); using local order",
                provider.name,
                provider.model,
            )
            return candidates[:top_k]
        return parse_ranking_response(ranking, candidates, top_k)
    except Exception as exc:
        logger.warning(
            "LLM rerank failed (%s/%s: %s); using local order",
            provider.name,
            provider.model,
            exc,
        )
        return candidates[:top_k]
