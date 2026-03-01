"""
Tool for executing LLM synthesis on a pre-built context package (Stage 2+3).

Accepts a serialized ContextPackage (from query_context) and runs
synthesize_context + execute to produce a QueryResponse.
"""

from __future__ import annotations

import json
from typing import Any, Dict, TYPE_CHECKING

from .base import BaseTool

if TYPE_CHECKING:
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

    def __init__(self, pipeline: "RetrievalPipeline") -> None:
        self.pipeline = pipeline

    def execute(self, args: Dict[str, Any]) -> str:
        from lsm.query.pipeline_types import ContextPackage, QueryRequest

        question = str(args.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")

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

        # Run synthesize_context to resolve labels/prompt, then execute
        package = self.pipeline.synthesize_context(package)
        response = self.pipeline.execute(package)

        return json.dumps(response.to_dict(), indent=2)
