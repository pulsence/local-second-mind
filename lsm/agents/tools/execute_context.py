"""
Tool for executing LLM synthesis on a pre-built context package (Stage 2+3).

Accepts a serialized ContextPackage (from query_context) and runs
synthesize_context + execute to produce a QueryResponse.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
    from lsm.config.models.agents import SandboxConfig
    from lsm.query.pipeline import RetrievalPipeline


class ExecuteContextTool(BaseTool):
    """Synthesize an answer from a previously retrieved context package."""

    name = "execute_context"
    description = (
        "Run LLM synthesis on a context package returned by query_context. "
        "Use this to produce a grounded answer after inspecting candidates."
    )
    risk_level = "network"
    needs_network = True
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to answer using the context.",
            },
            "context_block": {
                "type": "string",
                "description": "The formatted context text containing source chunks.",
            },
            "starting_prompt": {
                "type": "string",
                "description": "Synthesis instructions (system prompt).",
            },
            "mode": {
                "type": "string",
                "description": "Query mode name for synthesis instructions.",
            },
            "prior_response_id": {
                "type": "string",
                "description": "Response ID for server cache continuation.",
            },
            "conversation_id": {
                "type": "string",
                "description": "Conversation session identifier.",
            },
        },
        "required": ["question", "context_block"],
    }

    def __init__(
        self,
        pipeline: "RetrievalPipeline",
        sandbox_config: Optional["SandboxConfig"] = None,
    ) -> None:
        self.pipeline = pipeline
        self.sandbox_config = sandbox_config

    def execute(self, args: Dict[str, Any]) -> str:
        from lsm.query.pipeline_types import ContextPackage, QueryRequest
        from .mode_validation import validate_agent_mode

        question = str(args.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")

        mode = args.get("mode")
        error = validate_agent_mode(mode, self.pipeline.config, self.sandbox_config)
        if error:
            raise ValueError(error)

        context_block = str(args.get("context_block", "")).strip()
        if not context_block:
            raise ValueError("context_block is required")

        request = QueryRequest(
            question=question,
            mode=args.get("mode"),
            starting_prompt=args.get("starting_prompt"),
            conversation_id=args.get("conversation_id"),
            prior_response_id=args.get("prior_response_id"),
        )

        package = ContextPackage(
            request=request,
            context_block=context_block,
            starting_prompt=args.get("starting_prompt"),
            prior_response_id=args.get("prior_response_id"),
        )

        if package.candidates:
            package = self.pipeline.synthesize_context(package)
        else:
            if package.request.starting_prompt is not None:
                starting_prompt = package.request.starting_prompt
            elif package.prior_response_id or package.request.prior_response_id:
                starting_prompt = None
            else:
                if hasattr(self.pipeline, "_resolve_mode"):
                    mode_config = self.pipeline._resolve_mode(request)
                else:
                    mode_config = self.pipeline.config.get_mode_config(
                        getattr(request, "resolved_mode", None)
                    )
                starting_prompt = getattr(mode_config, "synthesis_instructions", None)
            package = ContextPackage(
                request=request,
                context_block=context_block,
                starting_prompt=starting_prompt,
                prior_response_id=args.get("prior_response_id"),
            )
        response = self.pipeline.execute(package)

        return json.dumps(response.to_dict(), indent=2)
