"""Savepoint-aware transaction context manager.

Provides a single ``transaction()`` context manager that transparently
uses ``BEGIN/COMMIT`` for top-level transactions and ``SAVEPOINT/RELEASE``
for nested calls, removing the need for per-provider state tracking.

Supports both SQLite (``sqlite3.Connection``) and PostgreSQL (``psycopg2``)
connections.  PostgreSQL always operates inside a transaction block, so the
context manager uses savepoints unconditionally on that backend.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from lsm.db.compat import execute, is_sqlite

_local = threading.local()


def _next_savepoint_id() -> int:
    """Return a monotonically increasing savepoint counter per thread."""
    counter = getattr(_local, "savepoint_counter", 0) + 1
    _local.savepoint_counter = counter
    return counter


def _in_transaction(conn: Any) -> bool:
    """Check whether *conn* is already inside a transaction.

    SQLite exposes ``conn.in_transaction``.  PostgreSQL (psycopg2) uses
    ``conn.status`` where ``STATUS_IN_TRANSACTION`` (2) and
    ``STATUS_INTRANS`` (2) indicate an active block.
    """
    if is_sqlite(conn):
        return bool(conn.in_transaction)
    # psycopg2
    try:
        import psycopg2.extensions as ext  # type: ignore[import-untyped]

        return conn.status == ext.STATUS_IN_TRANSACTION
    except (ImportError, AttributeError):
        # Fallback: assume inside a transaction (safe default for PG)
        return True


@contextmanager
def transaction(conn: Any) -> Iterator[None]:
    """Savepoint-aware transaction context manager.

    * If a transaction is already active, uses a named savepoint so the
      inner block can roll back independently.
    * Otherwise starts a top-level ``BEGIN … COMMIT`` block.

    Usage::

        with transaction(conn):
            execute(conn, "INSERT …")
            with transaction(conn):          # nested — uses savepoint
                execute(conn, "UPDATE …")
    """
    if _in_transaction(conn):
        savepoint = f"lsm_sp_{_next_savepoint_id()}"
        execute(conn, f"SAVEPOINT {savepoint}")
        try:
            yield
            execute(conn, f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            execute(conn, f"ROLLBACK TO SAVEPOINT {savepoint}")
            execute(conn, f"RELEASE SAVEPOINT {savepoint}")
            raise
        return

    if is_sqlite(conn):
        try:
            conn.execute("BEGIN")
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    else:
        # PostgreSQL: psycopg2 auto-opens a transaction on first statement.
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
