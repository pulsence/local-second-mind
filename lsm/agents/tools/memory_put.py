"""
Tool for proposing new memory candidates.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.agents.memory import BaseMemoryStore, Memory
from lsm.agents.memory.api import memory_put_candidate

from .base import BaseTool


class MemoryPutTool(BaseTool):
    """Create pending memory candidates for later promotion/rejection."""

    name = "memory_put"
    description = "Propose a persistent memory candidate for later approval."
    requires_permission = True
    risk_level = "writes_workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Memory key name."},
            "value": {"description": "JSON-serializable memory payload."},
            "type": {
                "type": "string",
                "description": "Memory type: pinned|project_fact|task_state|cache.",
            },
            "scope": {
                "type": "string",
                "description": "Memory scope: global|agent|project.",
            },
            "tags": {"type": "array", "description": "Optional memory tags."},
            "rationale": {
                "type": "string",
                "description": "Reason this memory candidate should be kept.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score from 0.0 to 1.0.",
            },
            "provenance": {
                "type": "string",
                "description": "Optional provenance string for audit history.",
            },
            "source_run_id": {
                "type": "string",
                "description": "Optional source run identifier.",
            },
        },
        "required": ["key", "value", "rationale"],
    }

    def __init__(self, store: BaseMemoryStore) -> None:
        self.store = store

    def execute(self, args: Dict[str, Any]) -> str:
        key = str(args.get("key", "")).strip()
        rationale = str(args.get("rationale", "")).strip()
        if not key:
            raise ValueError("key is required")
        if not rationale:
            raise ValueError("rationale is required")

        tags_raw = args.get("tags", [])
        tags = (
            [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            if isinstance(tags_raw, list)
            else []
        )
        memory = Memory(
            type=str(args.get("type", "project_fact")).strip().lower(),
            key=key,
            value=args.get("value"),
            scope=str(args.get("scope", "project")).strip().lower(),
            tags=tags,
            confidence=float(args.get("confidence", 1.0)),
            source_run_id=str(args.get("source_run_id", "tool-memory_put")).strip(),
        )
        memory.validate()

        provenance = str(args.get("provenance", "agent_tool")).strip() or "agent_tool"
        candidate_id = memory_put_candidate(
            self.store,
            memory,
            provenance=provenance,
            rationale=rationale,
        )
        payload = {
            "candidate_id": candidate_id,
            "memory_id": memory.id,
            "status": "pending",
            "memory": _serialize_memory(memory),
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)


def _serialize_memory(memory: Memory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "type": memory.type,
        "key": memory.key,
        "value": memory.value,
        "scope": memory.scope,
        "tags": list(memory.tags),
        "confidence": float(memory.confidence),
        "created_at": memory.created_at.isoformat(),
        "last_used_at": memory.last_used_at.isoformat(),
        "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
        "source_run_id": memory.source_run_id,
    }
