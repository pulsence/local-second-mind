"""
OpenAI LLM provider implementation.

Implements the BaseLLMProvider interface for OpenAI's API.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from openai import OpenAI, OpenAIError, RateLimitError

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider implementation.

    Uses OpenAI's Responses API for reranking and synthesis.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration with API key and model settings
        """
        self.config = config

        # Create OpenAI client
        if config.api_key:
            self.client = OpenAI(api_key=config.api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var

        logger.debug(f"Initialized OpenAI provider with model: {config.model}")

    @property
    def name(self) -> str:
        """Get provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Get current model."""
        return self.config.model

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        if self.config.api_key:
            return True
        return bool(os.getenv("OPENAI_API_KEY"))

    def _is_quota_error(self, error: Exception) -> bool:
        """Check if error is due to quota/rate limit."""
        if isinstance(error, RateLimitError):
            try:
                data = getattr(error, "response", None)
                if data is not None:
                    j = data.json()
                    return j.get("error", {}).get("code") == "insufficient_quota"
            except Exception:
                pass
            return True
        return isinstance(error, OpenAIError)

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using OpenAI LLM.

        Args:
            question: User's question
            candidates: List of candidate dicts with 'text' and 'metadata'
            k: Number of candidates to return
            **kwargs: Additional options (temperature, etc.)

        Returns:
            Reranked list of candidates (top k)
        """
        if not candidates:
            return []

        k = max(1, min(k, len(candidates)))

        # Prepare candidates for reranking
        items = []
        for i, cand in enumerate(candidates):
            text = cand.get("text", "")
            metadata = cand.get("metadata", {})

            # Truncate long text
            if len(text) > 1200:
                text = text[:1150] + "\n...[truncated]..."

            items.append({
                "index": i,
                "source_path": metadata.get("source_path", "unknown"),
                "source_name": metadata.get("source_name"),
                "chunk_index": metadata.get("chunk_index"),
                "ext": metadata.get("ext"),
                "distance": cand.get("distance"),
                "text": text,
            })

        # Build reranking prompt
        instructions = (
            "You are a retrieval reranker.\n"
            "Goal: rank the candidate passages by how useful they are for answering the user's question.\n"
            "Guidance:\n"
            "- Prefer passages that directly address the question.\n"
            "- Prefer specificity, definitions, arguments, or evidence over vague mentions.\n"
            "- If multiple passages are similar, rank the most comprehensive/precise first.\n"
            "- Do NOT hallucinate facts; you are only ranking.\n\n"
            "Output requirements:\n"
            "- Return STRICT JSON only, no markdown, no extra text.\n"
            "- Schema: {\"ranking\":[{\"index\":int,\"reason\":string}...]}\n"
            f"- Include at most {k} items.\n"
        )

        payload = {
            "question": question,
            "top_n": k,
            "candidates": items,
        }

        try:
            logger.debug(f"Requesting LLM rerank for {len(candidates)} candidates -> top {k}")

            resp = self.client.responses.create(
                model=self.config.model,
                reasoning={"effort": "low"},
                instructions=instructions,
                input=[{"role": "user", "content": json.dumps(payload)}],
                response_format={"type": "json_object"},
            )

            raw = (resp.output_text or "").strip()
            logger.debug(f"LLM rerank response length: {len(raw)} chars")

            # Parse ranking
            data = json.loads(raw)
            ranking = data.get("ranking", [])

            if not isinstance(ranking, list):
                logger.warning("Invalid ranking format, falling back to vector order")
                return candidates[:k]

            # Reconstruct candidates in ranked order
            chosen = []
            seen = set()

            for r in ranking:
                if not isinstance(r, dict) or "index" not in r:
                    continue
                idx = int(r["index"])
                if 0 <= idx < len(candidates) and idx not in seen:
                    chosen.append(candidates[idx])
                    seen.add(idx)
                if len(chosen) >= k:
                    break

            # Fill any gaps from original order
            if len(chosen) < k:
                for i, c in enumerate(candidates):
                    if i not in seen:
                        chosen.append(c)
                    if len(chosen) >= k:
                        break

            logger.info(f"LLM reranked {len(candidates)} → {len(chosen)} candidates")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM rerank response: {e}")
            return candidates[:k]

        except Exception as e:
            if self._is_quota_error(e):
                logger.warning(f"LLM rerank quota/rate limit hit: {e}")
            else:
                logger.warning(f"LLM rerank failed: {e}")
            return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        """
        Generate an answer with citations.

        Args:
            question: User's question
            context: Context block with [S1], [S2] citations
            mode: Query mode ('grounded' or 'insight')
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            Generated answer text with citations
        """
        # Build mode-specific instructions
        if mode == "insight":
            instructions = (
                "You are a research analyst. Analyze the provided sources to identify:\n"
                "- Recurring themes and patterns\n"
                "- Contradictions or tensions\n"
                "- Gaps or open questions\n"
                "- Evolution of ideas across documents\n\n"
                "Cite sources [S#] when referencing specific passages, but focus on\n"
                "synthesis across the corpus rather than answering narrow questions.\n"
                "Style: analytical, thematic, insightful."
            )
        else:  # grounded mode (default)
            instructions = (
                "Answer the user's question using ONLY the provided sources.\n"
                "Citation rules:\n"
                "- Whenever you make a claim supported by a source, cite inline like [S1] or [S2].\n"
                "- If multiple sources support a sentence, include multiple citations.\n"
                "- Do not fabricate citations.\n"
                "- If the sources are insufficient, say so and specify what is missing.\n"
                "Style: concise, structured, directly responsive."
            )

        user_content = (
            f"Question:\n{question}\n\n"
            f"Sources:\n{context}\n\n"
            "Write the answer with inline citations."
        )

        try:
            logger.debug(f"Requesting synthesis in {mode} mode")

            # Get optional parameters
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

            resp = self.client.responses.create(
                model=self.config.model,
                reasoning={"effort": "medium"},
                instructions=instructions,
                input=[{"role": "user", "content": user_content}],
                # Note: temperature and max_tokens may not be supported by responses API
                # Keeping them here for future compatibility
            )

            answer = (resp.output_text or "").strip()
            logger.info(f"Generated answer ({len(answer)} chars) in {mode} mode")
            return answer

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"Synthesis quota/rate limit hit: {e}")
                return self._fallback_answer(question, context)
            else:
                logger.error(f"Synthesis failed: {e}")
                return self._fallback_answer(question, context)

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        """
        Generate a fallback answer when API is unavailable.

        Args:
            question: User's question
            context: Context block
            max_chars: Maximum characters to include

        Returns:
            Fallback message with context snippet
        """
        snippet = context[:max_chars]
        if len(context) > max_chars:
            snippet += "\n...[truncated]..."

        return (
            f"[Offline mode: OpenAI API unavailable]\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{snippet}\n\n"
            f"Note: Unable to generate synthesized answer. "
            f"Please review the sources above directly."
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate cost for OpenAI API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD (rough estimate)
        """
        # Very rough estimates for GPT-4/GPT-5 pricing
        # These should be updated with actual pricing

        if "gpt-5" in self.config.model:
            # Placeholder pricing (actual pricing TBD)
            input_cost_per_1k = 0.01
            output_cost_per_1k = 0.03
        elif "gpt-4" in self.config.model:
            input_cost_per_1k = 0.01
            output_cost_per_1k = 0.03
        else:
            # Default estimate
            input_cost_per_1k = 0.001
            output_cost_per_1k = 0.002

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        return input_cost + output_cost
