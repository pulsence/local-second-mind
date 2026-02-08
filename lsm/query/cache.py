"""
Query result cache with TTL and LRU eviction.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class QueryCache:
    """
    In-memory query cache with TTL and LRU eviction.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive")
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()

    @staticmethod
    def build_key(
        query_text: str,
        mode: str,
        filters: Optional[Dict[str, Any]],
        k: int,
        k_rerank: int,
        conversation: Optional[str] = None,
    ) -> str:
        """
        Build a stable hash key for a query request.
        """
        payload = {
            "query_text": query_text or "",
            "mode": mode or "",
            "filters": filters or {},
            "k": int(k),
            "k_rerank": int(k_rerank),
            "conversation": conversation or "",
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        """
        Get cached value if present and not expired.
        """
        self._evict_expired()
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.time():
            self._store.pop(key, None)
            return None
        # LRU promotion
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """
        Store value and enforce LRU capacity.
        """
        expires_at = time.time() + self.ttl_seconds
        self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [key for key, entry in self._store.items() if entry.expires_at <= now]
        for key in expired:
            self._store.pop(key, None)
