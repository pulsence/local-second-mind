"""
Tool for querying promoted agent memories.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.agents.memory import BaseMemoryStore
from lsm.agents.memory.api import memory_search

from .base import BaseTool


class MemorySearchTool(BaseTool):
    """Search promoted memories through the configured memory backend."""

    name = "memory_search"
    description = "Search promoted memories by scope, tags, and memory type."
    risk_level = "read_only"
    input_schema = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "description": "Optional scope filter: global|agent|project.",
            },
            "tags": {"type": "array", "description": "Optional tag filter list."},
            "type": {
                "type": "string",
                "description": "Optional type filter: pinned|project_fact|task_state|cache.",
            },
            "limit": {"type": "integer", "description": "Maximum number of records."},
            "token_budget": {
                "type": "integer",
                "description": "Optional token-budget cap for returned records.",
            },
        },
        "required": [],
    }

    def __init__(self, store: BaseMemoryStore) -> None:
        self.store = store

    def execute(self, args: Dict[str, Any]) -> str:
        tags_raw = args.get("tags")
        tags = (
            [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            if isinstance(tags_raw, list)
            else None
        )
        memories = memory_search(
            self.store,
            scope=_optional_text(args.get("scope")),
            tags=tags,
            memory_type=_optional_text(args.get("type")),
            limit=max(1, int(args.get("limit", 10))),
            token_budget=(
                max(1, int(args["token_budget"]))
                if args.get("token_budget") is not None
                else None
            ),
            update_last_used=False,
        )
        payload = [
            {
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
            for memory in memories
        ]
        return json.dumps(payload, indent=2, ensure_ascii=True)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
