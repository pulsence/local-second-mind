"""Statistics caching for collection metadata analysis.

Caches the expensive full-scan metadata analysis to a JSON file alongside the
vector DB persist directory.  The cache is invalidated when the chunk count
changes (indicating ingest has run) or when the maximum age is exceeded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.logging import get_logger

logger = get_logger(__name__)


class StatsCache:
    """Cache for collection statistics.

    Stores computed statistics to avoid re-scanning the entire collection
    on every stats request.

    Args:
        cache_path: Path to the cache JSON file.
        max_age_seconds: Maximum age in seconds before cache is considered stale
            regardless of chunk count.  Default 3600 (1 hour).
    """

    def __init__(
        self,
        cache_path: Path,
        max_age_seconds: int = 3600,
    ) -> None:
        self._cache_path = cache_path
        self._max_age_seconds = max_age_seconds

    def load(self) -> Optional[Dict[str, Any]]:
        """Load cached stats from disk.

        Returns:
            Cache envelope ``{cached_at, chunk_count, stats}`` or ``None``
            if the file is missing or corrupt.
        """
        if not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "stats" not in data:
                return None
            return data
        except Exception:
            return None

    def save(self, stats: Dict[str, Any], chunk_count: int) -> None:
        """Save stats to disk with current chunk count and timestamp.

        Args:
            stats: Computed statistics dictionary.
            chunk_count: Current chunk count at time of computation.
        """
        envelope: Dict[str, Any] = {
            "cached_at": time.time(),
            "chunk_count": chunk_count,
            "stats": stats,
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_stale(self, current_count: int) -> bool:
        """Check whether the cache is stale.

        Stale when:
        - Cache file does not exist or is corrupt.
        - Chunk count differs from the cached count.
        - Cache age exceeds ``max_age_seconds``.

        Args:
            current_count: Current number of chunks in the collection.

        Returns:
            ``True`` if the cache should be recomputed.
        """
        data = self.load()
        if data is None:
            return True
        if data.get("chunk_count") != current_count:
            return True
        cached_at = data.get("cached_at", 0)
        if (time.time() - cached_at) > self._max_age_seconds:
            return True
        return False

    def get_if_fresh(self, current_count: int) -> Optional[Dict[str, Any]]:
        """Return cached stats if fresh, else ``None``.

        Args:
            current_count: Current number of chunks in the collection.

        Returns:
            The stats dictionary or ``None`` if the cache is stale.
        """
        if self.is_stale(current_count):
            return None
        data = self.load()
        if data is None:
            return None
        return data["stats"]

    def invalidate(self) -> None:
        """Delete the cache file."""
        try:
            if self._cache_path.exists():
                self._cache_path.unlink()
        except OSError as exc:
            logger.debug("Failed to delete stats cache %s: %s", self._cache_path, exc)
