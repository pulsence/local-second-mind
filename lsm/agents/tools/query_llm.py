"""
Tool for direct LLM prompting with cache-aware chaining support.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from lsm.config.models import LLMRegistryConfig
from lsm.providers.factory import create_provider
from lsm.query.context import format_user_content
from lsm.query.prompts import (
    SYNTHESIZE_GROUNDED_INSTRUCTIONS,
    SYNTHESIZE_INSIGHT_INSTRUCTIONS,
)

from .base import BaseTool


class QueryLLMTool(BaseTool):
    """Query an LLM service configured in `llms.services`."""

    name = "query_llm"
    description = "Run a direct prompt against a configured LLM service."
    tier = "normal"
    risk_level = "network"
    needs_network = True
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Prompt text to send."},
            "service": {
                "type": "string",
                "description": "Optional service name from llms.services (default: default).",
            },
            "tier": {
                "type": "string",
                "description": "Optional LLM tier name (quick/normal/complex).",
            },
            "context": {"type": "string", "description": "Optional context block."},
            "mode": {"type": "string", "description": "Synthesis mode (grounded or insight)."},
            "previous_response_id": {
                "type": "string",
                "description": "Response ID from a prior turn for server cache chaining.",
            },
            "prompt_cache_key": {
                "type": "string",
                "description": "Cache key for provider-side prompt caching.",
            },
            "prompt_cache_retention": {
                "type": "integer",
                "description": "Retention hint for prompt caching (seconds).",
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        llm_registry: LLMRegistryConfig,
        default_service: str = "default",
    ) -> None:
        self.llm_registry = llm_registry
        self.default_service = default_service

    def execute(self, args: Dict[str, Any]) -> str:
        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")

        service = str(args.get("service", "")).strip()
        tier = str(args.get("tier", "")).strip().lower()
        context = str(args.get("context", ""))
        mode = str(args.get("mode", "insight")).strip().lower() or "insight"

        previous_response_id: Optional[str] = args.get("previous_response_id")
        prompt_cache_key: Optional[str] = args.get("prompt_cache_key")
        prompt_cache_retention: Optional[int] = args.get("prompt_cache_retention")

        if service:
            llm_config = self.llm_registry.resolve_service(service)
        elif tier:
            llm_config = self.llm_registry.resolve_tier(tier)
        else:
            llm_config = self.llm_registry.resolve_service(self.default_service)
        provider = create_provider(llm_config)

        cache_kwargs: Dict[str, Any] = {}
        if previous_response_id:
            cache_kwargs["previous_response_id"] = previous_response_id
        if prompt_cache_key:
            cache_kwargs["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            cache_kwargs["prompt_cache_retention"] = prompt_cache_retention

        if context.strip():
            instructions = (
                SYNTHESIZE_INSIGHT_INSTRUCTIONS
                if mode == "insight"
                else SYNTHESIZE_GROUNDED_INSTRUCTIONS
            )
            answer = provider.send_message(
                input=format_user_content(prompt, context),
                instruction=instructions,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                **cache_kwargs,
            )
        else:
            answer = provider.send_message(
                input=prompt,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                **cache_kwargs,
            )

        response_id = getattr(provider, "last_response_id", None)

        output: Dict[str, Any] = {"answer": answer}
        if response_id:
            output["response_id"] = str(response_id)

        return json.dumps(output)
