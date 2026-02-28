"""Statistics caching for collection metadata analysis.

Supports DB-backed cache storage in ``lsm_stats_cache`` (preferred) with a
legacy JSON file fallback for non-sqlite flows.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lsm.db.connection import create_sqlite_connection, resolve_db_path
from lsm.logging import get_logger

logger = get_logger(__name__)


class StatsCache:
    """Cache for collection statistics."""

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        max_age_seconds: int = 3600,
        *,
        connection: Optional[sqlite3.Connection] = None,
        db_path: Optional[Path] = None,
        cache_key: str = "collection_stats",
    ) -> None:
        self._cache_path = cache_path
        self._max_age_seconds = int(max_age_seconds)
        self._cache_key = str(cache_key or "collection_stats")
        self._connection = connection
        self._owns_connection = False

        if self._connection is None and db_path is not None:
            resolved = resolve_db_path(Path(db_path))
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._connection = create_sqlite_connection(resolved)
            self._owns_connection = True

        if self._connection is not None:
            self._ensure_table()

    def close(self) -> None:
        if self._owns_connection and self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None
            self._owns_connection = False

    def __del__(self) -> None:
        self.close()

    def _ensure_table(self) -> None:
        if self._connection is None:
            return
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lsm_stats_cache (
                cache_key TEXT PRIMARY KEY,
                cached_at REAL NOT NULL,
                chunk_count INTEGER NOT NULL,
                stats_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_lsm_stats_cache_cached_at ON lsm_stats_cache(cached_at)"
        )
        self._connection.commit()

    def load(self) -> Optional[Dict[str, Any]]:
        """Load cached stats envelope."""
        if self._connection is not None:
            row = self._connection.execute(
                """
                SELECT cached_at, chunk_count, stats_json
                FROM lsm_stats_cache
                WHERE cache_key = ?
                """,
                (self._cache_key,),
            ).fetchone()
            if row is None:
                return None
            try:
                stats = json.loads(str(row["stats_json"]))
            except Exception:
                return None
            if not isinstance(stats, dict):
                return None
            return {
                "cached_at": float(row["cached_at"]),
                "chunk_count": int(row["chunk_count"]),
                "stats": stats,
            }

        if self._cache_path is None or not self._cache_path.exists():
            return None
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "stats" not in data:
                return None
            return data
        except Exception:
            return None

    def save(self, stats: Dict[str, Any], chunk_count: int) -> None:
        """Save stats envelope."""
        envelope: Dict[str, Any] = {
            "cached_at": time.time(),
            "chunk_count": int(chunk_count),
            "stats": stats,
        }
        if self._connection is not None:
            self._connection.execute(
                """
                INSERT INTO lsm_stats_cache(cache_key, cached_at, chunk_count, stats_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    cached_at = excluded.cached_at,
                    chunk_count = excluded.chunk_count,
                    stats_json = excluded.stats_json
                """,
                (
                    self._cache_key,
                    float(envelope["cached_at"]),
                    int(envelope["chunk_count"]),
                    json.dumps(stats, ensure_ascii=True),
                ),
            )
            self._connection.commit()
            return

        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def is_stale(self, current_count: int) -> bool:
        data = self.load()
        if data is None:
            return True
        if data.get("chunk_count") != current_count:
            return True
        cached_at = data.get("cached_at", 0.0)
        if (time.time() - float(cached_at)) > self._max_age_seconds:
            return True
        return False

    def get_if_fresh(self, current_count: int) -> Optional[Dict[str, Any]]:
        if self.is_stale(current_count):
            return None
        data = self.load()
        if data is None:
            return None
        return data["stats"]

    def invalidate(self) -> None:
        if self._connection is not None:
            self._connection.execute(
                "DELETE FROM lsm_stats_cache WHERE cache_key = ?",
                (self._cache_key,),
            )
            self._connection.commit()
            return
        try:
            if self._cache_path is not None and self._cache_path.exists():
                self._cache_path.unlink()
        except OSError as exc:
            logger.debug("Failed to delete stats cache %s: %s", self._cache_path, exc)
