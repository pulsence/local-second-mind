"""Caching helpers for remote provider results.

Preferred storage is ``lsm_remote_cache`` in the unified SQLite DB. Legacy
filesystem JSON cache is kept as a fallback for non-sqlite flows.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lsm.db.connection import create_sqlite_connection, resolve_db_path
from lsm.logging import get_logger
from lsm.paths import get_global_folder

logger = get_logger(__name__)


def _sanitize_provider_name(provider_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(provider_name).strip())
    return cleaned or "unknown_provider"


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _cache_path(provider_name: str, query: str, global_folder: Optional[str | Path]) -> Path:
    root = get_global_folder(global_folder)
    provider_dir = root / "Downloads" / _sanitize_provider_name(provider_name)
    provider_dir.mkdir(parents=True, exist_ok=True)
    return provider_dir / f"{_query_hash(query)}.json"


def _query_cache_key(provider_name: str, query: str) -> str:
    return f"query:{_sanitize_provider_name(provider_name)}:{_query_hash(query)}"


def _feed_cache_key(feed_url: str) -> str:
    return f"feed:rss:{_query_hash(feed_url)}"


def _open_cache_connection(
    *,
    db_connection: Optional[sqlite3.Connection],
    vectordb_path: Optional[str | Path],
) -> tuple[Optional[sqlite3.Connection], Optional[Path], bool]:
    if db_connection is not None:
        _ensure_remote_cache_table(db_connection)
        return db_connection, None, False
    if vectordb_path is None:
        return None, None, False
    db_path = resolve_db_path(Path(vectordb_path).expanduser())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_sqlite_connection(db_path)
    _ensure_remote_cache_table(conn)
    return conn, db_path, True


def _ensure_remote_cache_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lsm_remote_cache (
            cache_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_lsm_remote_cache_provider ON lsm_remote_cache(provider)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_lsm_remote_cache_expires_at ON lsm_remote_cache(expires_at)"
    )
    connection.commit()


def _load_db_payload(
    connection: sqlite3.Connection,
    cache_key: str,
) -> Optional[tuple[Dict[str, Any], Optional[datetime], Optional[datetime]]]:
    row = connection.execute(
        """
        SELECT response_json, created_at, expires_at
        FROM lsm_remote_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(str(row["response_json"]))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload, _parse_iso(row["created_at"]), _parse_iso(row["expires_at"])


def _save_db_payload(
    connection: sqlite3.Connection,
    *,
    cache_key: str,
    provider: str,
    payload: Dict[str, Any],
    cache_ttl_seconds: Optional[int],
) -> None:
    now = _now_utc()
    created_at = now.isoformat()
    expires_at: Optional[str] = None
    if cache_ttl_seconds is not None and int(cache_ttl_seconds) > 0:
        expires_at = (now + timedelta(seconds=int(cache_ttl_seconds))).isoformat()
    connection.execute(
        """
        INSERT INTO lsm_remote_cache(
            cache_key, provider, response_json, created_at, expires_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            provider = excluded.provider,
            response_json = excluded.response_json,
            created_at = excluded.created_at,
            expires_at = excluded.expires_at
        """,
        (
            cache_key,
            provider,
            json.dumps(payload, ensure_ascii=True),
            created_at,
            expires_at,
        ),
    )
    connection.commit()


def save_results(
    provider_name: str,
    query: str,
    results: List[Dict[str, Any]],
    global_folder: Optional[str | Path],
    *,
    db_connection: Optional[sqlite3.Connection] = None,
    vectordb_path: Optional[str | Path] = None,
    cache_ttl_seconds: Optional[int] = None,
) -> Path:
    """Save remote provider search results to cache."""
    payload = {
        "provider": provider_name,
        "query": query,
        "saved_at": int(time.time()),
        "results": results,
    }

    connection = None
    db_path: Optional[Path] = None
    should_close = False
    try:
        connection, db_path, should_close = _open_cache_connection(
            db_connection=db_connection,
            vectordb_path=vectordb_path,
        )
        if connection is not None:
            _save_db_payload(
                connection,
                cache_key=_query_cache_key(provider_name, query),
                provider=provider_name,
                payload=payload,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            return db_path if db_path is not None else Path("lsm.db")
    finally:
        if should_close and connection is not None:
            connection.close()

    cache_file = _cache_path(provider_name, query, global_folder)
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return cache_file


def load_cached_results(
    provider_name: str,
    query: str,
    global_folder: Optional[str | Path],
    max_age: int,
    *,
    db_connection: Optional[sqlite3.Connection] = None,
    vectordb_path: Optional[str | Path] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Load cached remote provider search results when fresh."""
    connection = None
    should_close = False
    try:
        connection, _, should_close = _open_cache_connection(
            db_connection=db_connection,
            vectordb_path=vectordb_path,
        )
        if connection is not None:
            loaded = _load_db_payload(connection, _query_cache_key(provider_name, query))
            if loaded is None:
                return None
            payload, created_at, expires_at = loaded

            now = _now_utc()
            if expires_at is not None and now > expires_at:
                connection.execute(
                    "DELETE FROM lsm_remote_cache WHERE cache_key = ?",
                    (_query_cache_key(provider_name, query),),
                )
                connection.commit()
                return None

            if expires_at is None:
                saved_at = payload.get("saved_at")
                if isinstance(saved_at, int):
                    if (int(time.time()) - saved_at) > int(max_age):
                        return None
                elif created_at is not None and (now - created_at).total_seconds() > int(max_age):
                    return None

            results = payload.get("results")
            return results if isinstance(results, list) else None
    finally:
        if should_close and connection is not None:
            connection.close()

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
    if age > int(max_age):
        return None
    return results


@dataclass
class FeedCache:
    """Cached RSS/Atom feed payload."""

    feed_url: str
    saved_at: int
    items: List[Dict[str, Any]]
    seen_ids: List[str]
    fresh: bool


def _feed_cache_path(feed_url: str, global_folder: Optional[str | Path]) -> Path:
    return _cache_path("rss", feed_url, global_folder)


def save_feed_cache(
    feed_url: str,
    items: List[Dict[str, Any]],
    seen_ids: List[str],
    global_folder: Optional[str | Path],
    *,
    db_connection: Optional[sqlite3.Connection] = None,
    vectordb_path: Optional[str | Path] = None,
    cache_ttl_seconds: Optional[int] = None,
) -> Path:
    """Save RSS/Atom feed cache."""
    payload = {
        "feed_url": feed_url,
        "saved_at": int(time.time()),
        "items": items,
        "seen_ids": [str(value) for value in seen_ids if str(value).strip()],
    }

    connection = None
    db_path: Optional[Path] = None
    should_close = False
    try:
        connection, db_path, should_close = _open_cache_connection(
            db_connection=db_connection,
            vectordb_path=vectordb_path,
        )
        if connection is not None:
            _save_db_payload(
                connection,
                cache_key=_feed_cache_key(feed_url),
                provider="rss",
                payload=payload,
                cache_ttl_seconds=cache_ttl_seconds,
            )
            return db_path if db_path is not None else Path("lsm.db")
    finally:
        if should_close and connection is not None:
            connection.close()

    cache_file = _feed_cache_path(feed_url, global_folder)
    cache_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return cache_file


def load_feed_cache(
    feed_url: str,
    global_folder: Optional[str | Path],
    max_age: int,
    *,
    db_connection: Optional[sqlite3.Connection] = None,
    vectordb_path: Optional[str | Path] = None,
) -> Optional[FeedCache]:
    """Load RSS/Atom feed cache when available."""
    connection = None
    should_close = False
    try:
        connection, _, should_close = _open_cache_connection(
            db_connection=db_connection,
            vectordb_path=vectordb_path,
        )
        if connection is not None:
            loaded = _load_db_payload(connection, _feed_cache_key(feed_url))
            if loaded is None:
                return None
            payload, created_at, expires_at = loaded

            saved_at = payload.get("saved_at")
            items = payload.get("items")
            seen_ids = payload.get("seen_ids") or []
            if not isinstance(saved_at, int):
                saved_at = int(created_at.timestamp()) if created_at is not None else int(time.time())
            if not isinstance(items, list):
                items = []
            if not isinstance(seen_ids, list):
                seen_ids = []

            now = _now_utc()
            if expires_at is not None:
                fresh = now <= expires_at
            elif created_at is not None:
                fresh = (now - created_at).total_seconds() <= int(max_age)
            else:
                fresh = (int(time.time()) - saved_at) <= int(max_age)

            return FeedCache(
                feed_url=feed_url,
                saved_at=saved_at,
                items=items,
                seen_ids=[str(value) for value in seen_ids if str(value).strip()],
                fresh=bool(fresh),
            )
    finally:
        if should_close and connection is not None:
            connection.close()

    cache_file = _feed_cache_path(feed_url, global_folder)
    if not cache_file.exists():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to read feed cache file '{cache_file}': {exc}")
        return None

    saved_at = payload.get("saved_at")
    items = payload.get("items")
    seen_ids = payload.get("seen_ids") or []
    if not isinstance(saved_at, int):
        return None
    if not isinstance(items, list):
        items = []
    if not isinstance(seen_ids, list):
        seen_ids = []

    age = int(time.time()) - saved_at
    fresh = age <= int(max_age)
    return FeedCache(
        feed_url=feed_url,
        saved_at=saved_at,
        items=items,
        seen_ids=[str(value) for value in seen_ids if str(value).strip()],
        fresh=fresh,
    )
