"""Connection management for the unified database.

Provides a single source of truth for creating and resolving database
connections so every subsystem gets consistent PRAGMA configuration
(WAL journal, foreign keys, busy timeout, ``sqlite3.Row`` factory).

Includes a unified ``resolve_connection()`` context manager that yields
a raw DB-API connection from any provider instance, regardless of
backend.

Ownership semantics
-------------------
- **SQLite**: yields the provider's persistent ``.connection`` reference.
  The caller must **not** close it — the provider owns the lifecycle.
- **PostgreSQL**: yields a pool-borrowed connection via ``._get_conn()``.
  The connection is returned to the pool when the context exits.

Transaction semantics
---------------------
Both backends operate with autocommit **off** by default.  Callers
should use ``compat.commit()`` explicitly when they want to persist
changes.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple, Union


def create_vectordb_provider(config: Any) -> Any:
    """Create a vector DB provider via the canonical db-level entrypoint."""
    from lsm.db import create_vectordb_provider as _create_vectordb_provider

    return _create_vectordb_provider(config)


def create_sqlite_connection(path: Path) -> sqlite3.Connection:
    """Create a SQLite connection with standard LSM pragmas.

    Applies WAL journal mode, foreign keys ON, busy_timeout 5000,
    ``check_same_thread=False``, and ``row_factory=sqlite3.Row``.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def resolve_db_path(path: Path) -> Path:
    """Resolve a database file path.

    If *path* already ends with ``.db``, return as-is.
    Otherwise append ``lsm.db`` (treating *path* as a directory).
    """
    if str(path).lower().endswith(".db"):
        return path
    return path / "lsm.db"


def resolve_vectordb_provider_name(
    vectordb: Any,
) -> str:
    """Return the canonical provider name (``'sqlite'`` or ``'postgresql'``)."""
    # Avoid circular import — check duck-type first.
    if _is_provider_instance(vectordb):
        return str(getattr(vectordb, "name", "") or "").strip().lower()
    # Assume DBConfig
    return str(getattr(vectordb, "provider", "") or "sqlite").strip().lower()


def resolve_sqlite_connection(
    vectordb: Any,
) -> Tuple[sqlite3.Connection, bool]:
    """Extract or create a SQLite connection from a vectordb config/provider.

    Returns ``(connection, owns_connection)`` where *owns_connection* is
    ``True`` when the function instantiated a new provider (caller should
    close it).
    """
    if _is_provider_instance(vectordb):
        if resolve_vectordb_provider_name(vectordb) != "sqlite":
            raise ValueError(
                "SQLite connection resolution requires vectordb provider='sqlite' "
                "or a SQLite vector provider instance."
            )
        connection = getattr(vectordb, "connection", None)
        if not isinstance(connection, sqlite3.Connection):
            raise ValueError(
                "SQLite vector provider does not expose a valid SQLite connection."
            )
        return connection, False

    if resolve_vectordb_provider_name(vectordb) != "sqlite":
        raise ValueError(
            "SQLite connection resolution requires vectordb.provider='sqlite'."
        )

    provider = create_vectordb_provider(vectordb)
    connection = getattr(provider, "connection", None)
    if not isinstance(connection, sqlite3.Connection):
        raise ValueError(
            "SQLite vector provider did not expose a valid SQLite connection."
        )
    return connection, True


def resolve_postgres_connection_factory(
    vectordb: Any,
) -> Optional[Callable[[], Any]]:
    """Extract a PostgreSQL connection factory from a provider instance.

    Returns ``None`` when the provider is not PostgreSQL or does not
    expose a ``_get_conn`` callable.
    """
    if not _is_provider_instance(vectordb):
        return None
    if resolve_vectordb_provider_name(vectordb) != "postgresql":
        return None
    get_conn = getattr(vectordb, "_get_conn", None)
    if callable(get_conn):
        return get_conn
    return None


@contextmanager
def resolve_connection(provider: Any) -> Iterator[Any]:
    """Yield a raw DB-API connection from a vector-DB provider instance.

    Works for both SQLite and PostgreSQL providers.

    - **SQLite**: yields ``provider.connection`` (persistent — do NOT close).
    - **PostgreSQL**: yields ``provider._get_conn()`` (pool-borrowed —
      returned on context exit automatically by the pool's semantics).

    Raises:
        ValueError: If *provider* does not expose a SQL connection.

    Usage::

        with resolve_connection(provider) as conn:
            compat.execute(conn, "SELECT 1")
    """
    name = resolve_vectordb_provider_name(provider)

    if name == "sqlite":
        connection = getattr(provider, "connection", None)
        if connection is None:
            raise ValueError(
                "SQLite provider does not expose a .connection attribute."
            )
        yield connection
        return

    if name == "postgresql":
        get_conn = getattr(provider, "_get_conn", None)
        if not callable(get_conn):
            raise ValueError(
                "PostgreSQL provider does not expose a _get_conn() callable."
            )
        conn = get_conn()
        try:
            yield conn
        finally:
            # Pool-borrowed connections are returned automatically
            # when the caller's transaction is complete.
            pass
        return

    raise ValueError(
        f"Provider '{name}' does not expose SQL access. "
        f"Supported providers: sqlite, postgresql."
    )


def _is_provider_instance(vectordb: Any) -> bool:
    """Check whether *vectordb* is a provider instance (vs. a config)."""
    try:
        from lsm.vectordb.base import BaseVectorDBProvider

        if isinstance(vectordb, BaseVectorDBProvider):
            return True
    except ImportError:
        pass
    return hasattr(vectordb, "name") and hasattr(vectordb, "config")
