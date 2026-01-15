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


_UNSUPPORTED_PARAMS_BY_MODEL: dict[str, set[str]] = {}


def _is_param_unsupported(model: str, param: str) -> bool:
    return param in _UNSUPPORTED_PARAMS_BY_MODEL.get(model, set())


def _should_send_param(model: str, param: str) -> bool:
    return not _is_param_unsupported(model, param)


def _mark_param_unsupported(model: str, param: str) -> None:
    _UNSUPPORTED_PARAMS_BY_MODEL.setdefault(model, set()).add(param)


def _is_unsupported_param_error(error: Exception, param: str) -> bool:
    message = str(error)
    return (
        f"Unsupported parameter: '{param}'" in message
        or f"Unsupported parameter: \"{param}\"" in message
        or f"unexpected keyword argument '{param}'" in message
    )


def _model_supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


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

        super().__init__()
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

    def list_models(self) -> List[str]:
        """List models available to the current API key."""
        if not self.is_available():
            return []

        try:
            res = self.client.models.list()
            ids: List[str] = []
            for model in getattr(res, "data", []) or []:
                model_id = getattr(model, "id", None)
                if isinstance(model_id, str):
                    ids.append(model_id)
            ids.sort()
            return ids
        except Exception as e:
            logger.debug(f"Failed to list OpenAI models: {e}")
            return []

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

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

    def _call_responses(self, request_args: Dict[str, Any], action: str):
        return self._with_retry(
            lambda: self.client.responses.create(**request_args),
            action,
            retry_on=self._is_retryable_error,
        )

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

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": json.dumps(payload)}],
            }

            if _should_send_param(self.config.model, "text"):
                request_args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "rerank_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "ranking": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "index": {"type": "integer"},
                                            "reason": {"type": "string"}
                                        },
                                        "required": ["index", "reason"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["ranking"],
                            "additionalProperties": False
                        }
                    }
                }

            try:
                resp = self._call_responses(request_args, "rerank")
            except Exception as e:
                if _is_unsupported_param_error(e, "text"):
                    logger.warning("Model does not support text.format for structured output; retrying without it.")
                    _mark_param_unsupported(self.config.model, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "rerank")
                else:
                    raise

            raw = (resp.output_text or "").strip()
            logger.debug(f"LLM rerank response length: {len(raw)} chars")

            # Parse ranking
            data = json.loads(raw)
            ranking = data.get("ranking", [])

            if not isinstance(ranking, list):
                logger.warning("Invalid ranking format, falling back to vector order")
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
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

            self._record_success("rerank")
            logger.info(f"LLM reranked {len(candidates)} → {len(chosen)} candidates")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            if self._is_quota_error(e):
                logger.warning(f"LLM rerank quota/rate limit hit: {e}")
            else:
                logger.warning(f"LLM rerank failed: {e}")
            self._record_failure(e, "rerank")
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

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "medium"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if (
                temperature is not None
                and _model_supports_temperature(self.config.model)
                and _should_send_param(self.config.model, "temperature")
            ):
                request_args["temperature"] = temperature

            try:
                resp = self._call_responses(request_args, "synthesize")
            except Exception as e:
                if _is_unsupported_param_error(e, "temperature"):
                    logger.warning("Model does not support temperature; retrying without it.")
                    _mark_param_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "synthesize")
                else:
                    raise

            answer = (resp.output_text or "").strip()
            self._record_success("synthesize")
            logger.info(f"Generated answer ({len(answer)} chars) in {mode} mode")
            return answer

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"Synthesis quota/rate limit hit: {e}")
                self._record_failure(e, "synthesize")
                return self._fallback_answer(question, context)
            else:
                logger.error(f"Synthesis failed: {e}")
                self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate relevant tags for a text chunk using OpenAI LLM.

        Args:
            text: Text content to tag
            num_tags: Number of tags to generate (default: 3)
            existing_tags: Optional list of existing tags to consider for consistency
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            List of generated tag strings

        Raises:
            Exception: If tag generation fails
        """
        # Build instructions
        existing_context = ""
        if existing_tags:
            existing_context = f"\n\nExisting tags in this knowledge base: {', '.join(existing_tags[:20])}"

        instructions = f"""You are a helpful assistant that generates concise, relevant tags for text content.

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

        user_content = f"Text:\n{text[:2000]}"

        try:
            logger.debug(f"Requesting tag generation for text of length {len(text)}")

            # Get optional parameters
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

            request_args = {
                "model": self.config.model,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if temperature is not None and _should_send_param(self.config.model, "temperature"):
                request_args["temperature"] = temperature

            # Responses API uses text.format for structured output, not response_format
            if _should_send_param(self.config.model, "text"):
                request_args["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "tags_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["tags"],
                            "additionalProperties": False
                        }
                    }
                }

            try:
                resp = self._call_responses(request_args, "generate_tags")
            except Exception as e:
                # Handle unsupported parameters
                if _is_unsupported_param_error(e, "temperature"):
                    logger.warning("Model does not support temperature for tagging; retrying without it.")
                    _mark_param_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "generate_tags")
                elif _is_unsupported_param_error(e, "text"):
                    logger.warning("Model does not support text.format for structured output; retrying without it.")
                    _mark_param_unsupported(self.config.model, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "generate_tags")
                else:
                    raise

            # Extract tags from response
            content = (resp.output_text or "").strip()
            logger.debug(f"LLM tagging response length: {len(content)} chars")

            if not content:
                logger.warning("LLM response was empty")
                self._record_failure(ValueError("Empty tag response"), "generate_tags")
                return []

            # Try to parse as JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and isinstance(data.get("tags"), list):
                    tags = data["tags"]
                    if tags and all(isinstance(t, str) for t in tags):
                        tags = [t.lower().strip() for t in tags if t.strip()]
                        logger.info(f"Generated {len(tags)} tags for text")
                        self._record_success("generate_tags")
                        return tags[:num_tags]
                elif isinstance(data, list):
                    tags = data
                    if tags and all(isinstance(t, str) for t in tags):
                        tags = [t.lower().strip() for t in tags if t.strip()]
                        logger.info(f"Generated {len(tags)} tags for text")
                        self._record_success("generate_tags")
                        return tags[:num_tags]

                logger.warning(f"JSON parsed but unexpected structure: {data}")

            except json.JSONDecodeError as e:
                logger.debug(f"Initial JSON parse failed: {e}")

                # Try to extract JSON object or array from the response
                # Look for {"tags": [...]} pattern
                obj_start = content.find("{")
                obj_end = content.rfind("}")
                if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                    try:
                        json_str = content[obj_start:obj_end + 1]
                        data = json.loads(json_str)
                        if isinstance(data, dict) and isinstance(data.get("tags"), list):
                            tags = data["tags"]
                            if tags and all(isinstance(t, str) for t in tags):
                                tags = [t.lower().strip() for t in tags if t.strip()]
                                logger.info(f"Generated {len(tags)} tags from object extraction")
                                self._record_success("generate_tags")
                                return tags[:num_tags]
                    except json.JSONDecodeError:
                        pass

                # Try to extract just the array  [...] pattern
                arr_start = content.find("[")
                arr_end = content.rfind("]")
                if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                    try:
                        json_str = content[arr_start:arr_end + 1]
                        data = json.loads(json_str)
                        if isinstance(data, list) and all(isinstance(t, str) for t in data):
                            tags = [t.lower().strip() for t in data if t.strip()]
                            logger.info(f"Generated {len(tags)} tags from array extraction")
                            self._record_success("generate_tags")
                            return tags[:num_tags]
                    except json.JSONDecodeError:
                        pass

                # Last resort: treat as comma-separated list (only if no JSON structure found)
                if "{" not in content and "[" not in content:
                    tags = [t.strip().lower() for t in content.split(",")]
                    tags = [t for t in tags if t]
                    if tags:
                        logger.info(f"Generated {len(tags)} tags from comma-separated text")
                        self._record_success("generate_tags")
                        return tags[:num_tags]

                logger.warning(f"Could not parse tags from response: {content!r}")
                self._record_failure(ValueError("Failed to parse tag response"), "generate_tags")
                return []

            logger.error(f"Could not extract tags from LLM response: {content}")
            self._record_failure(ValueError("Empty tag response"), "generate_tags")
            return []

        except Exception as e:
            logger.error(f"Failed to generate tags: {e}")
            self._record_failure(e, "generate_tags")
            raise

    def stream_synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ):
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
        else:
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

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        request_args = {
            "model": self.config.model,
            "reasoning": {"effort": "medium"},
            "instructions": instructions,
            "input": [{"role": "user", "content": user_content}],
            "max_output_tokens": max_tokens,
        }

        if (
            temperature is not None
            and _model_supports_temperature(self.config.model)
            and _should_send_param(self.config.model, "temperature")
        ):
            request_args["temperature"] = temperature

        try:
            try:
                stream = self.client.responses.create(**request_args, stream=True)
            except Exception as e:
                if _is_unsupported_param_error(e, "temperature"):
                    logger.warning("Model does not support temperature; retrying without it.")
                    _mark_param_unsupported(self.config.model, "temperature")
                    request_args.pop("temperature", None)
                    stream = self.client.responses.create(**request_args, stream=True)
                else:
                    raise

            emitted = False
            for event in stream:
                event_type = getattr(event, "type", None)
                if event_type is None and isinstance(event, dict):
                    event_type = event.get("type")
                if event_type in {"response.output_text.delta", "response.output_text"}:
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    if delta:
                        emitted = True
                        yield delta
                elif event_type == "response.completed":
                    break

            self._record_success("stream_synthesize")
            if not emitted:
                logger.warning("OpenAI stream completed without text output")

        except Exception as e:
            if self._is_quota_error(e):
                logger.error(f"Streaming quota/rate limit hit: {e}")
            else:
                logger.error(f"Streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

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
