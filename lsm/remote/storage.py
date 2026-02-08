"""
Disk caching helpers for remote provider results.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.logging import get_logger
from lsm.paths import get_global_folder

logger = get_logger(__name__)


def _sanitize_provider_name(provider_name: str) -> str:
    """Create a filesystem-safe provider folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(provider_name).strip())
    return cleaned or "unknown_provider"


def _query_hash(query: str) -> str:
    """Create a stable hash for query-specific cache file names."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]


def _cache_path(provider_name: str, query: str, global_folder: Optional[str | Path]) -> Path:
    """Resolve cache file path for a provider/query pair."""
    root = get_global_folder(global_folder)
    provider_dir = root / "Downloads" / _sanitize_provider_name(provider_name)
    provider_dir.mkdir(parents=True, exist_ok=True)
    return provider_dir / f"{_query_hash(query)}.json"


def save_results(
    provider_name: str,
    query: str,
    results: List[Dict[str, Any]],
    global_folder: Optional[str | Path],
) -> Path:
    """
    Save remote provider results to disk cache.
    """
    cache_file = _cache_path(provider_name, query, global_folder)
    payload = {
        "provider": provider_name,
        "query": query,
        "saved_at": int(time.time()),
        "results": results,
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return cache_file


def load_cached_results(
    provider_name: str,
    query: str,
    global_folder: Optional[str | Path],
    max_age: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Load cached remote provider results when available and fresh.
    """
    cache_file = _cache_path(provider_name, query, global_folder)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read remote cache file '{cache_file}': {exc}")
        return None

    saved_at = payload.get("saved_at")
    results = payload.get("results")
    if not isinstance(saved_at, int) or not isinstance(results, list):
        return None

    age = int(time.time()) - saved_at
    if age > max_age:
        return None

    return results
