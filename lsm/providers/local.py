"""
Local model provider implementation (Ollama).
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

import requests

from lsm.config.models import LLMConfig
from lsm.logging import get_logger
from .base import BaseLLMProvider
from .helpers import (
    RERANK_INSTRUCTIONS,
    format_user_content,
    generate_fallback_answer,
    get_synthesis_instructions,
    get_tag_instructions,
    parse_json_payload,
    parse_ranking_response,
    prepare_candidates_for_rerank,
)

logger = get_logger(__name__)


class LocalProvider(BaseLLMProvider):
    """
    Local model provider using Ollama's HTTP API.

    Configuration:
        provider: local
        model: llama2
        base_url: http://localhost:11434
        temperature: 0.7
        max_tokens: 2000
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (
            config.base_url
            or os.getenv("OLLAMA_BASE_URL")
            or "http://localhost:11434"
        ).rstrip("/")
        super().__init__()
        logger.debug(f"Initialized Local provider with model: {config.model}")

    @property
    def name(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self.config.model

    def is_available(self) -> bool:
        return bool(self.base_url)

    def health_check(self) -> Dict[str, Any]:
        data = super().health_check()
        data["base_url"] = self.base_url
        return data

    def _is_retryable_error(self, error: Exception) -> bool:
        return isinstance(error, requests.exceptions.RequestException)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def _chat(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        def _call() -> Dict[str, Any]:
            return self._post("/api/chat", payload)

        resp = self._with_retry(_call, "chat", retry_on=self._is_retryable_error)
        message = resp.get("message", {}) if isinstance(resp, dict) else {}
        return (message.get("content") or "").strip()

    def _chat_stream(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        url = f"{self.base_url}/api/chat"
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = data.get("message", {}) if isinstance(data, dict) else {}
            content = message.get("content")
            if content:
                yield content
            if data.get("done"):
                break

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

        items = prepare_candidates_for_rerank(candidates)
        instructions = RERANK_INSTRUCTIONS.format(k=k)

        payload = {
            "question": question,
            "top_n": k,
            "candidates": items,
        }

        try:
            raw = self._chat(instructions, json.dumps(payload), temperature=0.2, max_tokens=400)
            data = parse_json_payload(raw)
            ranking = data.get("ranking", []) if isinstance(data, dict) else None
            if not isinstance(ranking, list):
                self._record_failure(ValueError("Invalid rerank response"), "rerank")
                return candidates[:k]

            chosen = parse_ranking_response(ranking, candidates, k)

            self._record_success("rerank")
            return chosen

        except Exception as e:
            logger.warning(f"Local rerank failed: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

    def synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ) -> str:
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            answer = self._chat(instructions, user_content, temperature=temperature, max_tokens=max_tokens)
            self._record_success("synthesize")
            return answer
        except Exception as e:
            logger.error(f"Local synthesis failed: {e}")
            self._record_failure(e, "synthesize")
            return self._fallback_answer(question, context)

    def generate_tags(
        self,
        text: str,
        num_tags: int = 3,
        existing_tags: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        instructions = get_tag_instructions(num_tags, existing_tags)

        user_content = f"Text:\n{text[:2000]}"
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

        try:
            raw = self._chat(instructions, user_content, temperature=temperature, max_tokens=max_tokens)
            data = parse_json_payload(raw)
            tags = data.get("tags", []) if isinstance(data, dict) else None
            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Local tag generation failed: {e}")
            self._record_failure(e, "generate_tags")
            raise

    def stream_synthesize(
        self,
        question: str,
        context: str,
        mode: str = "grounded",
        **kwargs
    ):
        instructions = get_synthesis_instructions(mode)
        user_content = format_user_content(question, context)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            for chunk in self._chat_stream(
                instructions,
                user_content,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                if chunk:
                    yield chunk
            self._record_success("stream_synthesize")
        except Exception as e:
            logger.error(f"Local streaming synthesis failed: {e}")
            self._record_failure(e, "stream_synthesize")
            raise

    def get_model_pricing(self) -> Optional[Dict[str, float]]:
        """Local models are free -- always return zero-cost pricing."""
        return {"input": 0.0, "output": 0.0}

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        return generate_fallback_answer(question, context, "Local model", max_chars=max_chars)
