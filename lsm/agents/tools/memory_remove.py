"""
Tool for deleting existing memories.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.agents.memory import BaseMemoryStore

from .base import BaseTool


class MemoryRemoveTool(BaseTool):
    """Delete a memory record by ID."""

    name = "memory_remove"
    description = "Remove a memory and its candidate history by memory ID."
    requires_permission = True
    risk_level = "writes_workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Memory ID to remove.",
            }
        },
        "required": ["memory_id"],
    }

    def __init__(self, store: BaseMemoryStore) -> None:
        self.store = store

    def execute(self, args: Dict[str, Any]) -> str:
        memory_id = str(args.get("memory_id", "")).strip()
        if not memory_id:
            raise ValueError("memory_id is required")
        self.store.delete(memory_id)
        return json.dumps(
            {
                "operation": "remove_memory",
                "memory_id": memory_id,
                "status": "deleted",
            },
            indent=2,
            ensure_ascii=True,
        )
