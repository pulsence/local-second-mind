"""
Build standing memory context blocks for agent prompt injection.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .api import memory_expire, memory_search
from .models import Memory, now_utc
from .store import BaseMemoryStore


@dataclass
class MemoryContextPayload:
    """
    Built memory prompt payload.
    """

    text: str
    memories: List[Memory]


class MemoryContextBuilder:
    """
    Build "Standing Context" prompt blocks from persisted memory.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        *,
        default_limit: int = 8,
        default_token_budget: int = 1200,
    ) -> None:
        self.store = store
        self.default_limit = max(1, int(default_limit))
        self.default_token_budget = max(1, int(default_token_budget))

    def build(
        self,
        *,
        agent_name: str,
        topic: str = "",
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> str:
        """
        Return standing context text to inject into the agent system prompt.
        """
        payload = self.build_payload(
            agent_name=agent_name,
            topic=topic,
            tags=tags,
            memory_type=memory_type,
            limit=limit,
            token_budget=token_budget,
        )
        return payload.text

    def build_payload(
        self,
        *,
        agent_name: str,
        topic: str = "",
        tags: Optional[Sequence[str]] = None,
        memory_type: Optional[str] = None,
        limit: Optional[int] = None,
        token_budget: Optional[int] = None,
    ) -> MemoryContextPayload:
        """
        Build a full memory payload (text + selected memories).
        """
        requested_limit = max(1, int(limit or self.default_limit))
        requested_budget = token_budget or self.default_token_budget
        topic_text = str(topic or "").strip()
        normalized_tags = self._normalize_tags(tags)
        topic_tags = self._extract_topic_tags(topic_text)
        all_tags = normalized_tags
        # Avoid over-constraining search with many topic tags because store tag filters
        # are AND-based. For topic-derived filtering, use a single strongest tag.
        if not all_tags and topic_tags:
            all_tags = [topic_tags[0]]

        # Cleanup first so stale memory never enters standing context.
        memory_expire(self.store)

        selected: list[Memory] = []
        seen_ids: set[str] = set()
        remaining_budget = requested_budget
        for scope in ("agent", "project", "global"):
            if len(selected) >= requested_limit:
                break
            scope_limit = requested_limit - len(selected)
            if scope_limit <= 0:
                break
            scope_memories = memory_search(
                self.store,
                scope=scope,
                tags=all_tags or None,
                memory_type=memory_type,
                limit=max(scope_limit * 2, scope_limit),
                token_budget=remaining_budget,
                update_last_used=False,
            )
            for memory in scope_memories:
                if memory.id in seen_ids:
                    continue
                selected.append(memory)
                seen_ids.add(memory.id)
                if remaining_budget is not None and remaining_budget > 0:
                    remaining_budget = max(0, int(remaining_budget) - self._estimate_tokens(memory))
                if len(selected) >= requested_limit:
                    break
                if remaining_budget == 0:
                    break
            if remaining_budget == 0:
                break

        if not selected:
            return MemoryContextPayload(text="", memories=[])

        # Update usage only for memories actually injected.
        self.store.mark_used([memory.id for memory in selected], used_at=now_utc())
        block = self._format_block(agent_name=agent_name, topic=topic_text, memories=selected)
        return MemoryContextPayload(text=block, memories=selected)

    @staticmethod
    def _normalize_tags(tags: Optional[Sequence[str]]) -> list[str]:
        if not tags:
            return []
        normalized = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _extract_topic_tags(topic: str) -> list[str]:
        if not topic:
            return []
        tokens = re.findall(r"[a-zA-Z0-9_]{4,}", topic.lower())
        stop_words = {
            "about",
            "from",
            "that",
            "this",
            "with",
            "where",
            "which",
            "when",
            "what",
            "into",
            "would",
            "could",
            "should",
            "does",
            "role",
            "understand",
        }
        filtered = [token for token in tokens if token not in stop_words]
        return list(dict.fromkeys(filtered[:8]))

    @staticmethod
    def _estimate_tokens(memory: Memory) -> int:
        payload = f"{memory.key} {memory.value} {' '.join(memory.tags)} {memory.scope} {memory.type}"
        return max(1, len(payload) // 4)

    @staticmethod
    def _format_block(*, agent_name: str, topic: str, memories: Sequence[Memory]) -> str:
        lines = ["Standing Context (Memory)"]
        lines.append(f"Agent: {agent_name}")
        if topic:
            lines.append(f"Topic: {topic}")
        lines.append("")

        for index, memory in enumerate(memories, start=1):
            tags_text = ", ".join(memory.tags) if memory.tags else "-"
            value_text = MemoryContextBuilder._serialize_value(memory.value)
            lines.append(
                f"{index}. key={memory.key} | type={memory.type} | scope={memory.scope} | confidence={memory.confidence:.2f}"
            )
            lines.append(f"   tags={tags_text}")
            lines.append(f"   value={value_text}")
        return "\n".join(lines).strip()

    @staticmethod
    def _serialize_value(value: object) -> str:
        try:
            encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            encoded = str(value)
        if len(encoded) > 300:
            return encoded[:297] + "..."
        return encoded
