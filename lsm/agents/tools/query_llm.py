"""
Tool for direct LLM prompting.
"""

from __future__ import annotations

from typing import Any, Dict

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
        if service:
            llm_config = self.llm_registry.resolve_service(service)
        elif tier:
            llm_config = self.llm_registry.resolve_tier(tier)
        else:
            llm_config = self.llm_registry.resolve_service(self.default_service)
        provider = create_provider(llm_config)
        if context.strip():
            instructions = (
                SYNTHESIZE_INSIGHT_INSTRUCTIONS
                if mode == "insight"
                else SYNTHESIZE_GROUNDED_INSTRUCTIONS
            )
            return provider.send_message(
                input=format_user_content(prompt, context),
                instruction=instructions,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
        return provider.send_message(
            input=prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )
