"""
Tool for proposing new memory candidates.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from lsm.agents.memory import BaseMemoryStore, Memory
from lsm.agents.memory.api import memory_promote, memory_put_candidate

from .base import BaseTool


class MemoryPutTool(BaseTool):
    """Create pending memory candidates or update existing memories."""

    name = "memory_put"
    description = "Propose a persistent memory candidate for later approval."
    requires_permission = True
    risk_level = "writes_workspace"
    input_schema = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "Existing memory ID to update in-place.",
            },
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
        "required": [],
    }

    def __init__(self, store: BaseMemoryStore) -> None:
        self.store = store

    def execute(self, args: Dict[str, Any]) -> str:
        memory_id = str(args.get("memory_id", "")).strip()
        if memory_id:
            return self._execute_update(memory_id, args)
        return self._execute_create(args)

    def _execute_create(self, args: Dict[str, Any]) -> str:
        key = str(args.get("key", "")).strip()
        if not key:
            raise ValueError("key is required")
        if "value" not in args:
            raise ValueError("value is required for new memory")

        rationale = str(args.get("rationale", "")).strip() or "Proposed by memory_put tool."
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
            "operation": "create_candidate",
            "candidate_id": candidate_id,
            "memory_id": memory.id,
            "status": "pending",
            "memory": _serialize_memory(memory),
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _execute_update(self, memory_id: str, args: Dict[str, Any]) -> str:
        existing = self.store.get(memory_id)

        tags_raw = args.get("tags")
        tags = (
            [str(tag).strip() for tag in tags_raw if str(tag).strip()]
            if isinstance(tags_raw, list)
            else list(existing.tags)
        )
        updated = Memory(
            id=existing.id,
            type=_coalesce_text(args.get("type"), existing.type).lower(),
            key=_coalesce_text(args.get("key"), existing.key),
            value=args.get("value", existing.value),
            scope=_coalesce_text(args.get("scope"), existing.scope).lower(),
            tags=tags,
            confidence=float(args.get("confidence", existing.confidence)),
            created_at=existing.created_at,
            last_used_at=existing.last_used_at,
            expires_at=existing.expires_at,
            source_run_id=_coalesce_text(args.get("source_run_id"), existing.source_run_id),
        )
        updated.validate()
        rationale = str(args.get("rationale", "")).strip() or "Updated existing memory."
        provenance = str(args.get("provenance", "agent_tool")).strip() or "agent_tool"

        candidate_id = self._replace_memory(existing, updated, provenance, rationale)
        payload = {
            "operation": "update_memory",
            "candidate_id": candidate_id,
            "memory_id": updated.id,
            "status": "promoted",
            "memory": _serialize_memory(updated),
        }
        return json.dumps(payload, indent=2, ensure_ascii=True)

    def _replace_memory(
        self,
        existing: Memory,
        updated: Memory,
        provenance: str,
        rationale: str,
    ) -> str:
        self.store.delete(existing.id)
        try:
            candidate_id = memory_put_candidate(
                self.store,
                updated,
                provenance=provenance,
                rationale=rationale,
            )
            memory_promote(self.store, candidate_id)
            return candidate_id
        except Exception:
            restore_candidate_id = memory_put_candidate(
                self.store,
                existing,
                provenance="memory_put_restore",
                rationale="Automatic restore after failed update.",
            )
            memory_promote(self.store, restore_candidate_id)
            raise


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


def _coalesce_text(value: Any, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or str(fallback).strip()
