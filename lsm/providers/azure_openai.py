"""
Azure OpenAI provider implementation.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional

from openai import AzureOpenAI

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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI provider.

    Configuration:
        provider: azure_openai
        model: gpt-35-turbo
        api_key: ${AZURE_OPENAI_API_KEY}
        endpoint: https://your-resource.openai.azure.com/
        api_version: 2023-05-15
        deployment_name: gpt-35-turbo
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = config.api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = (
            config.deployment_name
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            or config.model
        )
        self.client = AzureOpenAI(
            api_key=config.api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )
        super().__init__()
        logger.debug(f"Initialized Azure OpenAI provider with deployment: {self.deployment_name}")

    @property
    def name(self) -> str:
        return "azure_openai"

    @property
    def model(self) -> str:
        return self.deployment_name

    def is_available(self) -> bool:
        has_key = bool(self.config.api_key or os.getenv("AZURE_OPENAI_API_KEY"))
        return has_key and bool(self.endpoint) and bool(self.api_version)

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
            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": json.dumps(payload)}],
            }

            if _should_send_param(self.deployment_name, "text"):
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
                    logger.warning("Deployment does not support text.format; retrying without it.")
                    _mark_param_unsupported(self.deployment_name, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "rerank")
                else:
                    raise

            raw = (resp.output_text or "").strip()
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

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Azure rerank response: {e}")
            self._record_failure(e, "rerank")
            return candidates[:k]

        except Exception as e:
            logger.warning(f"Azure rerank failed: {e}")
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
            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "medium"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if (
                temperature is not None
                and _model_supports_temperature(self.deployment_name)
                and _should_send_param(self.deployment_name, "temperature")
            ):
                request_args["temperature"] = temperature

            try:
                resp = self._call_responses(request_args, "synthesize")
            except Exception as e:
                if _is_unsupported_param_error(e, "temperature"):
                    logger.warning("Deployment does not support temperature; retrying without it.")
                    _mark_param_unsupported(self.deployment_name, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "synthesize")
                else:
                    raise

            answer = (resp.output_text or "").strip()
            self._record_success("synthesize")
            return answer

        except Exception as e:
            logger.error(f"Azure synthesis failed: {e}")
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

        try:
            temperature = kwargs.get("temperature", self.config.temperature)
            max_tokens = min(kwargs.get("max_tokens", self.config.max_tokens), 200)

            request_args = {
                "model": self.deployment_name,
                "reasoning": {"effort": "low"},
                "instructions": instructions,
                "input": [{"role": "user", "content": user_content}],
                "max_output_tokens": max_tokens,
            }

            if temperature is not None and _should_send_param(self.deployment_name, "temperature"):
                request_args["temperature"] = temperature

            if _should_send_param(self.deployment_name, "text"):
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
                if _is_unsupported_param_error(e, "temperature"):
                    logger.warning("Deployment does not support temperature for tagging; retrying without it.")
                    _mark_param_unsupported(self.deployment_name, "temperature")
                    request_args.pop("temperature", None)
                    resp = self._call_responses(request_args, "generate_tags")
                elif _is_unsupported_param_error(e, "text"):
                    logger.warning("Deployment does not support text.format; retrying without it.")
                    _mark_param_unsupported(self.deployment_name, "text")
                    request_args.pop("text", None)
                    resp = self._call_responses(request_args, "generate_tags")
                else:
                    raise

            content = (resp.output_text or "").strip()
            data = json.loads(content)
            tags = data.get("tags", [])

            if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                cleaned = [t.lower().strip() for t in tags if t.strip()]
                self._record_success("generate_tags")
                return cleaned[:num_tags]

            raise ValueError(f"Unexpected tag structure: {data}")

        except Exception as e:
            logger.error(f"Azure tag generation failed: {e}")
            self._record_failure(e, "generate_tags")
            raise

    def _fallback_answer(self, question: str, context: str, max_chars: int = 1200) -> str:
        snippet = context[:max_chars]
        if len(context) > max_chars:
            snippet += "\n...[truncated]..."
        return (
            "[Offline mode: Azure OpenAI unavailable]\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{snippet}\n\n"
            "Note: Unable to generate synthesized answer. "
            "Please review the sources above directly."
        )
