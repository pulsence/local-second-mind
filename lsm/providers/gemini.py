"""
Google Gemini provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types as genai_types

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger
from .base import BaseLLMProvider

logger = get_logger(__name__)


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini provider.

    Configuration:
        provider: gemini
        model: gemini-1.5-pro
        api_key: ${GOOGLE_API_KEY}
        temperature: 0.7
        max_tokens: 2000
    """

    PRICING_PER_1M = {
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    }

    def __init__(self, config: LLMConfig):
        self.config = config
        self._api_key = config.api_key or os.getenv("GOOGLE_API_KEY")
        self._client: Optional[genai.Client] = None
        super().__init__()
        logger.debug(f"Initialized Gemini provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _ensure_client(self) -> genai.Client:
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. "
                    "Set it in config or as environment variable."
                )
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _build_config(self, temperature: float, max_tokens: int):
        return genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        name = error.__class__.__name__
        return name in {
            "ResourceExhausted",
            "ServiceUnavailable",
            "DeadlineExceeded",
            "InternalServerError",
        }

    def _generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        client = self._ensure_client()

        def _call() -> Any:
            config = self._build_config(temperature, max_tokens)
            return client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

        resp = self._with_retry(_call, "generate_content", retry_on=self._is_retryable_error)
        text = getattr(resp, "text", None)
        if not text:
            return ""
        return text.strip()

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
            "- Schema: {\"ranking\":[{\"index\":int,\"reason\":string}...]}\n"
            f"- Include at most {k} items.\n"
        )

        payload = {
            "question": question,
            "top_n": k,
            "candidates": items,
        }

        try:
            prompt = f"{instructions}\n\n{json.dumps(payload)}"
            raw = self._generate(prompt, temperature=0.2, max_tokens=400)
            data = json.loads(raw)
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

        except Exception as e:
            logger.warning(f"Gemini rerank failed: {e}")
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
            prompt = f"{instructions}\n\n{user_content}"
            answer = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Gemini synthesis failed: {e}")
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
- Schema: {{"tags":["tag1","tag2","tag3"]}}
- Include exactly {num_tags} tags.
"""

        user_content = f"Text:\n{text[:2000]}"
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

        try:
            prompt = f"{instructions}\n\n{user_content}"
            raw = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
            data = json.loads(raw)
            tags = data.get("tags", [])
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Gemini tag generation failed: {e}")
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
            client = self._ensure_client()
            config = self._build_config(temperature, max_tokens)
            prompt = f"{instructions}\n\n{user_content}"
            stream = client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=config,
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text

            self._record_success("stream_synthesize")

        except Exception as e:
            logger.error(f"Gemini streaming synthesis failed: {e}")
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
            "[Offline mode: Gemini API unavailable]\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{snippet}\n\n"
            "Note: Unable to generate synthesized answer. "
            "Please review the sources above directly."
        )
