"""Savepoint-aware transaction context manager for SQLite.

Provides a single ``transaction()`` context manager that transparently
uses ``BEGIN/COMMIT`` for top-level transactions and ``SAVEPOINT/RELEASE``
for nested calls, removing the need for per-provider state tracking.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

_local = threading.local()


def _next_savepoint_id() -> int:
    """Return a monotonically increasing savepoint counter per thread."""
    counter = getattr(_local, "savepoint_counter", 0) + 1
    _local.savepoint_counter = counter
    return counter


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Savepoint-aware transaction context manager.

    * If a transaction is already active (``conn.in_transaction``), uses a
      named savepoint so the inner block can roll back independently.
    * Otherwise starts a top-level ``BEGIN … COMMIT`` block.

    Usage::

        with transaction(conn):
            conn.execute("INSERT …")
            with transaction(conn):          # nested — uses savepoint
                conn.execute("UPDATE …")
    """
    if conn.in_transaction:
        savepoint = f"lsm_sp_{_next_savepoint_id()}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        return

    try:
        conn.execute("BEGIN")
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise
