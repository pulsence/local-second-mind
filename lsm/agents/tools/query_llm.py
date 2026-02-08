"""
Tool for direct LLM prompting.
"""

from __future__ import annotations

from typing import Any, Dict

from lsm.config.models import LLMRegistryConfig
from lsm.providers.factory import create_provider

from .base import BaseTool


class QueryLLMTool(BaseTool):
    """Query an LLM service configured in `llms.services`."""

    name = "query_llm"
    description = "Run a direct prompt against a configured LLM service."
    input_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Prompt text to send."},
            "service": {
                "type": "string",
                "description": "Optional service name from llms.services (default: default).",
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

        service = str(args.get("service", self.default_service)).strip() or self.default_service
        context = str(args.get("context", ""))
        mode = str(args.get("mode", "insight")).strip().lower() or "insight"
        llm_config = self.llm_registry.resolve_service(service)
        provider = create_provider(llm_config)
        return provider.synthesize(prompt, context, mode=mode)

