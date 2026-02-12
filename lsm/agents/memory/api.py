"""
Memory API helpers and ranking logic.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Sequence

from .models import Memory, now_utc
from .store import BaseMemoryStore


def memory_put_candidate(
    store: BaseMemoryStore,
    memory: Memory,
    provenance: str,
    rationale: str,
) -> str:
    """
    Store a pending memory candidate and return candidate ID.
    """
    return store.put_candidate(memory=memory, provenance=provenance, rationale=rationale)


def memory_promote(store: BaseMemoryStore, candidate_id: str) -> Memory:
    """
    Promote a candidate memory and return the promoted memory.
    """
    return store.promote(candidate_id)


def memory_expire(store: BaseMemoryStore) -> int:
    """
    Expire memory records according to TTL and return removed count.
    """
    return store.expire()


def memory_search(
    store: BaseMemoryStore,
    scope: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    memory_type: Optional[str] = None,
    limit: int = 20,
    token_budget: Optional[int] = None,
    *,
    update_last_used: bool = False,
    pin_weight: float = 1.5,
) -> List[Memory]:
    """
    Search memories with rank scoring (recency + confidence + pin weighting).

    Args:
        store: Memory store backend.
        scope: Optional scope filter.
        tags: Optional required tags.
        memory_type: Optional memory type filter.
        limit: Max memories to return.
        token_budget: Optional token budget cap for returned memories.
        update_last_used: Whether to update `last_used_at` for returned items.
        pin_weight: Additional score bonus for pinned memories.

    Returns:
        Ranked memory list.
    """
    target_limit = max(1, int(limit))
    # Pull a broader candidate set, then apply rank scoring and token-budget capping.
    candidates = store.search(
        scope=scope,
        tags=tags,
        memory_type=memory_type,
        limit=max(target_limit * 3, target_limit),
        token_budget=None,
    )

    now = now_utc()
    ranked = sorted(
        candidates,
        key=lambda item: _memory_rank_score(item, reference=now, pin_weight=pin_weight),
        reverse=True,
    )

    selected: list[Memory] = []
    consumed_tokens = 0
    for memory in ranked:
        estimated_tokens = _estimate_tokens(memory)
        if token_budget is not None and token_budget > 0:
            if consumed_tokens + estimated_tokens > int(token_budget):
                continue
            consumed_tokens += estimated_tokens
        selected.append(memory)
        if len(selected) >= target_limit:
            break

    if update_last_used and selected:
        store.mark_used([memory.id for memory in selected], used_at=now)
    return selected


def _memory_rank_score(memory: Memory, *, reference: datetime, pin_weight: float) -> float:
    """
    Compute ranking score from recency, confidence, and pin weighting.
    """
    delta_seconds = max(0.0, (reference - memory.last_used_at).total_seconds())
    age_days = delta_seconds / 86_400.0
    recency_score = 1.0 / (1.0 + age_days)
    confidence_score = max(0.0, min(1.0, float(memory.confidence)))
    pinned_bonus = float(pin_weight) if memory.type == "pinned" else 0.0
    return pinned_bonus + (0.65 * recency_score) + (0.35 * confidence_score)


def _estimate_tokens(memory: Memory) -> int:
    """
    Approximate memory token footprint using a simple 4-char heuristic.
    """
    value_text = str(memory.value)
    payload = f"{memory.key} {value_text} {' '.join(memory.tags)} {memory.scope} {memory.type}"
    return max(1, len(payload) // 4)

