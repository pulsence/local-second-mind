"""
Shared utilities and prompt templates for LLM providers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


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

SYNTHESIZE_GROUNDED_INSTRUCTIONS = """Answer the user's question using ONLY the provided sources.
Citation rules:
- Whenever you make a claim supported by a source, cite inline like [S1] or [S2].
- If multiple sources support a sentence, include multiple citations.
- Do not fabricate citations.
- If the sources are insufficient, say so and specify what is missing.
Style: concise, structured, directly responsive.
"""

SYNTHESIZE_INSIGHT_INSTRUCTIONS = """You are a research analyst. Analyze the provided sources to identify:
- Recurring themes and patterns
- Contradictions or tensions
- Gaps or open questions
- Evolution of ideas across documents

Cite sources [S#] when referencing specific passages, but focus on
synthesis across the corpus rather than answering narrow questions.
Style: analytical, thematic, insightful.
"""

TAG_GENERATION_TEMPLATE = """You are a helpful assistant that generates concise, relevant tags for text content.

Analyze the following text and generate {num_tags} relevant tags.

Guidelines:
- Tags should be concise (1-3 words)
- Tags should be specific to the content
- Tags should help with organization and retrieval
- Use lowercase
- Separate multi-word tags with hyphens (e.g., "machine-learning")
{existing_context}

Output requirements:
- Return STRICT JSON only, no markdown, no extra text.
- Schema: {{"tags":["tag1","tag2","tag3"]}}
- Include exactly {num_tags} tags.
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

TAGS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["tags"],
    "additionalProperties": False,
}


def prepare_candidates_for_rerank(
    candidates: List[Dict[str, Any]],
    max_text_length: int = 1200,
) -> List[Dict[str, Any]]:
    """Prepare rerank candidate payload with truncation and source metadata."""
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
    """Convert ranking payload from model response into ordered candidates."""
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


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences around content."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_json_payload(raw: str) -> Optional[Any]:
    """Parse JSON payload from plain text or markdown-wrapped text."""
    cleaned = strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


def generate_fallback_answer(
    question: str,
    context: str,
    provider_name: str,
    max_chars: int = 1200,
) -> str:
    """Generate fallback answer when a provider call fails."""
    snippet = context[:max_chars]
    if len(context) > max_chars:
        snippet += "\n...[truncated]..."

    return (
        f"[Offline mode: {provider_name} unavailable]\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{snippet}\n\n"
        "Note: Unable to generate synthesized answer. "
        "Please review the sources above directly."
    )


def format_user_content(question: str, context: str) -> str:
    """Build user content block for synthesis prompts."""
    return (
        f"Question:\n{question}\n\n"
        f"Sources:\n{context}\n\n"
        "Write the answer with inline citations."
    )


def get_synthesis_instructions(mode: str) -> str:
    """Return mode-specific synthesis instruction block."""
    return (
        SYNTHESIZE_INSIGHT_INSTRUCTIONS
        if mode == "insight"
        else SYNTHESIZE_GROUNDED_INSTRUCTIONS
    )


def get_tag_instructions(num_tags: int, existing_tags: Optional[List[str]] = None) -> str:
    """Return tag generation instruction block."""
    existing_context = ""
    if existing_tags:
        existing_context = (
            f"\n\nExisting tags in this knowledge base: {', '.join(existing_tags[:20])}"
        )
    return TAG_GENERATION_TEMPLATE.format(
        num_tags=num_tags,
        existing_context=existing_context,
    )


class UnsupportedParamTracker:
    """Track unsupported model parameters and detection logic."""

    def __init__(self) -> None:
        self._unsupported: Dict[str, Set[str]] = {}

    def mark_unsupported(self, model: str, param: str) -> None:
        self._unsupported.setdefault(model, set()).add(param)

    def should_send(self, model: str, param: str) -> bool:
        return param not in self._unsupported.get(model, set())

    def is_unsupported_error(self, error: Exception, param: str) -> bool:
        message = str(error)
        return (
            f"Unsupported parameter: '{param}'" in message
            or f"Unsupported parameter: \"{param}\"" in message
            or f"unexpected keyword argument '{param}'" in message
        )
