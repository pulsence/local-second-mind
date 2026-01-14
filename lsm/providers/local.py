"""
Local model provider implementation (Ollama).
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

import requests

from lsm.config.models import LLMConfig
from lsm.cli.logging import get_logger
from .base import BaseLLMProvider

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
            raw = self._chat(instructions, json.dumps(payload), temperature=0.2, max_tokens=400)
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
            raw = self._chat(instructions, user_content, temperature=temperature, max_tokens=max_tokens)
            data = json.loads(raw)
            tags = data.get("tags", [])
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

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        snippet = context[:max_chars]
        if len(context) > max_chars:
            snippet += "\n...[truncated]..."
        return (
            "[Offline mode: Local model unavailable]\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{snippet}\n\n"
            "Note: Unable to generate synthesized answer. "
            "Please review the sources above directly."
        )
