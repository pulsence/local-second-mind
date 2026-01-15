"""
Anthropic Claude provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from anthropic import Anthropic

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude provider.

    Configuration:
        provider: anthropic
        model: claude-3-5-sonnet-20241022
        api_key: ${ANTHROPIC_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    PRICING_PER_1M = {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Optional[Anthropic] = None
        super().__init__()
        logger.debug(f"Initialized Anthropic provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> List[str]:
        if not self.is_available():
            return []

        try:
            res = self.client.models.list(limit=1000)
            items = getattr(res, "data", None)
            if items is None:
                items = list(res)
            ids: List[str] = []
            for item in items or []:
                model_id = getattr(item, "id", None) or getattr(item, "name", None)
                if isinstance(model_id, str) and model_id:
                    ids.append(model_id)
            ids.sort()
            return ids
        except Exception as e:
            logger.debug(f"Failed to list Anthropic models: {e}")
            return []

    @property
    def client(self) -> Anthropic:
        if self._client is None:
            if not self.is_available():
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Set it in config or as environment variable."
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _extract_text(self, response: Any) -> str:
        parts = []
        for part in getattr(response, "content", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        return (getattr(response, "content", "") or "").strip()

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "RateLimitError",
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def _parse_json_payload(self, raw: str) -> Any:
        cleaned = self._strip_code_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                return json.loads(cleaned[obj_start:obj_end + 1])
            except json.JSONDecodeError:
                pass

        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                return json.loads(cleaned[arr_start:arr_end + 1])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Failed to parse JSON response", cleaned, 0)

    def rerank(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        k: int,
        **kwargs
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        k = max(1, min(k, len(candidates)))

        items = []
        for i, cand in enumerate(candidates):
            text = cand.get("text", "")
            metadata = cand.get("metadata", {})
            if len(text) > 1200:
                text = text[:1150] + "\n...[truncated]..."
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
            "- Do not include code fences or commentary.\n"
            "- Schema: {\"ranking\":[{\"index\":int,\"reason\":string}...]}\n"
            f"- Include at most {k} items.\n"
        )

        payload = {
            "question": question,
            "top_n": k,
            "candidates": items,
        }

        try:
            logger.debug(f"Requesting Claude rerank for {len(candidates)} candidates -> top {k}")

            def _call(system_prompt: str) -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=400,
                    temperature=0.2,
                    system=system_prompt,
                    messages=[{"role": "user", "content": json.dumps(payload)}],
                )

            resp = self._with_retry(
                lambda: _call(instructions),
                "rerank",
                retry_on=self._is_retryable_error,
            )
            raw = self._extract_text(resp)
            if not raw:
                raise json.JSONDecodeError("Empty rerank response", raw, 0)

            data = self._parse_json_payload(raw)
            ranking = data.get("ranking", [])
            if not isinstance(ranking, list):
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
                return candidates[:k]

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

            if len(chosen) < k:
                for i, c in enumerate(candidates):
                    if i not in seen:
                        chosen.append(c)
                    if len(chosen) >= k:
                        break

            self._record_success("rerank")
            return chosen

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Claude rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            logger.warning(f"Claude rerank failed: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
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

        try:
            logger.debug(f"Requesting Claude synthesis in {mode} mode")

            def _call() -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )

            resp = self._with_retry(_call, "synthesize", retry_on=self._is_retryable_error)
            answer = self._extract_text(resp)
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Claude synthesis failed: {e}")
            self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
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
- Do not include code fences or commentary.
- Schema: {{"tags":["tag1","tag2","tag3"]}}
- Include exactly {num_tags} tags.
"""

        user_content = f"Text:\n{text[:2000]}"

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

        try:
            logger.debug(f"Requesting Claude tag generation for text of length {len(text)}")

            def _call() -> Any:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )

            resp = self._with_retry(_call, "generate_tags", retry_on=self._is_retryable_error)
            content = self._extract_text(resp)
            if not content:
                raise json.JSONDecodeError("Empty tag response", content, 0)
            data = self._parse_json_payload(content)
            tags = data.get("tags", [])

            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Claude tag generation failed: {e}")
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

        try:
            try:
                stream = self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                )
                with stream as s:
                    for text in s.text_stream:
                        if text:
                            yield text
            except AttributeError:
                stream = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=instructions,
                    messages=[{"role": "user", "content": user_content}],
                    stream=True,
                )
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type is None and isinstance(event, dict):
                        event_type = event.get("type")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta is None and isinstance(event, dict):
                            delta = event.get("delta")
                        text = getattr(delta, "text", None)
                        if text is None and isinstance(delta, dict):
                            text = delta.get("text")
                        if text:
                            yield text

            self._record_success("stream_synthesize")

        except Exception as e:
            logger.error(f"Claude streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        rates = self.PRICING_PER_1M.get(self.model)
        if not rates:
            return None
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        snippet = context[:max_chars]
        if len(context) > max_chars:
            snippet += "\n...[truncated]..."
        return (
            "[Offline mode: Anthropic API unavailable]\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{snippet}\n\n"
            "Note: Unable to generate synthesized answer. "
            "Please review the sources above directly."
        )
