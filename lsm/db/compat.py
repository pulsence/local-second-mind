"""Cross-backend SQL compatibility layer.

Provides dialect-aware helpers so that DB modules can operate on both
SQLite (``sqlite3.Connection``) and PostgreSQL (``psycopg2`` connection)
without reimplementing dialect detection, placeholder conversion, or
row normalisation.

PostgreSQL dependencies are imported lazily — this module is safe to
import in SQLite-only environments without ``psycopg2`` installed.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ------------------------------------------------------------------
# Exception normalisation
# ------------------------------------------------------------------

class DBOperationalError(RuntimeError):
    """Normalised operational error that wraps backend-specific exceptions."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))


@contextmanager
def db_error(*extra_exc_types: type) -> Iterator[None]:
    """Context manager that catches backend DB errors and re-raises as
    :class:`DBOperationalError`.

    Always catches ``sqlite3.OperationalError``.  When ``psycopg2`` is
    available, also catches ``psycopg2.Error``.  Additional exception
    types can be supplied via *extra_exc_types*.

    Usage::

        with db_error():
            conn.execute("SELECT ...")
    """
    exc_types: list[type] = [sqlite3.OperationalError, sqlite3.DatabaseError]
    try:
        import psycopg2  # type: ignore[import-untyped]
        exc_types.append(psycopg2.Error)
    except ImportError:
        pass
    exc_types.extend(extra_exc_types)
    try:
        yield
    except tuple(exc_types) as exc:
        raise DBOperationalError(exc) from exc


# ------------------------------------------------------------------
# Dialect detection
# ------------------------------------------------------------------

def dialect(conn: Any) -> str:
    """Return ``"sqlite"`` or ``"postgresql"`` for a DB-API connection."""
    if isinstance(conn, sqlite3.Connection):
        return "sqlite"
    return "postgresql"


def is_sqlite(conn: Any) -> bool:
    """Return ``True`` when *conn* is a SQLite connection."""
    return isinstance(conn, sqlite3.Connection)


def is_postgres(conn: Any) -> bool:
    """Return ``True`` when *conn* is **not** a SQLite connection."""
    return not isinstance(conn, sqlite3.Connection)


# ------------------------------------------------------------------
# Placeholder and query conversion
# ------------------------------------------------------------------

def convert_placeholders(query: str, conn: Any) -> str:
    """Replace ``?`` with ``%s`` when *conn* is PostgreSQL."""
    if is_postgres(conn):
        return query.replace("?", "%s")
    return query


# ------------------------------------------------------------------
# SQL identifier validation
# ------------------------------------------------------------------

def safe_identifier(value: str) -> str:
    """Validate and return a SQL identifier.

    Raises ``ValueError`` for strings that are not safe to interpolate
    into SQL statements.
    """
    candidate = str(value).strip()
    if not _VALID_IDENTIFIER.match(candidate):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return candidate


# ------------------------------------------------------------------
# Execution helpers
# ------------------------------------------------------------------

def execute(conn: Any, query: str, params: Iterable[Any] = ()) -> Any:
    """Execute *query* with dialect-aware placeholder conversion.

    Returns the cursor (or cursor-like object) for further fetching.
    """
    sql = convert_placeholders(query, conn)
    param_tuple = tuple(params)
    if is_sqlite(conn):
        return conn.execute(sql, param_tuple)
    # psycopg2: conn may or may not have .execute(); use cursor
    cur = conn.cursor()
    cur.execute(sql, param_tuple)
    return cur


def executemany(conn: Any, query: str, params_seq: Iterable[Iterable[Any]]) -> Any:
    """Execute *query* with multiple parameter sets."""
    sql = convert_placeholders(query, conn)
    rows = [tuple(p) for p in params_seq]
    if is_sqlite(conn):
        return conn.executemany(sql, rows)
    cur = conn.cursor()
    cur.executemany(sql, rows)
    return cur


def execute_ddl_script(conn: Any, sql: str) -> None:
    """Split and execute multi-statement DDL.

    SQLite has ``executescript()`` but PostgreSQL does not — this helper
    splits on ``;`` and executes each non-empty statement individually.
    """
    if is_sqlite(conn):
        conn.executescript(sql)
        return
    cur = conn.cursor()
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)
    conn.commit()


# ------------------------------------------------------------------
# Fetch helpers
# ------------------------------------------------------------------

def fetchone(conn: Any, query: str, params: Iterable[Any] = ()) -> Optional[Any]:
    """Execute *query* and return a single row (or ``None``)."""
    cursor = execute(conn, query, params)
    return cursor.fetchone()


def fetchall(conn: Any, query: str, params: Iterable[Any] = ()) -> List[Any]:
    """Execute *query* and return all rows."""
    cursor = execute(conn, query, params)
    return cursor.fetchall()


# ------------------------------------------------------------------
# Row normalisation
# ------------------------------------------------------------------

def row_to_dict(row: Any, columns: Sequence[str] | None = None) -> Dict[str, Any]:
    """Normalise a row (``sqlite3.Row``, tuple, or mapping) to a ``dict``.

    When *columns* is ``None``, the function attempts to derive column
    names from the row object itself (works for ``sqlite3.Row`` and
    mapping types).  For plain tuples *columns* must be provided.
    """
    if row is None:
        return {}
    if isinstance(row, sqlite3.Row):
        return {k: row[k] for k in row.keys()}
    if isinstance(row, Mapping):
        return dict(row)
    if columns is not None:
        return dict(zip(columns, row))
    raise TypeError(
        f"Cannot convert row of type {type(row).__name__} to dict without column names"
    )


def _cursor_columns(cursor: Any) -> List[str]:
    """Extract column names from a DB-API cursor description."""
    return [item[0] for item in (cursor.description or [])]


def fetch_rows_as_dicts(
    conn: Any, query: str, params: Iterable[Any] = ()
) -> List[Dict[str, Any]]:
    """Execute *query* and return all rows as dicts."""
    cursor = execute(conn, query, params)
    columns = _cursor_columns(cursor)
    return [row_to_dict(row, columns) for row in cursor.fetchall()]


# ------------------------------------------------------------------
# Transaction / commit
# ------------------------------------------------------------------

def commit(conn: Any) -> None:
    """Commit the current transaction."""
    conn.commit()


# ------------------------------------------------------------------
# Table introspection
# ------------------------------------------------------------------

def table_exists(conn: Any, table_name: str) -> bool:
    """Check whether *table_name* exists in the database."""
    safe_name = safe_identifier(table_name)
    if is_sqlite(conn):
        row = execute(
            conn,
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (safe_name,),
        ).fetchone()
        return row is not None
    row = execute(conn, "SELECT to_regclass(%s)", (safe_name,)).fetchone()
    return bool(row and row[0])


def count_rows(conn: Any, table_name: str) -> int:
    """Return the number of rows in *table_name*."""
    safe_name = safe_identifier(table_name)
    if not table_exists(conn, safe_name):
        return 0
    row = execute(conn, f"SELECT COUNT(*) FROM {safe_name}").fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


# ------------------------------------------------------------------
# Insert / Upsert
# ------------------------------------------------------------------

def insert_returning_id(conn: Any, query: str, params: Iterable[Any] = ()) -> int:
    """Execute an INSERT and return the auto-generated primary key.

    For SQLite, uses ``cursor.lastrowid``.  For PostgreSQL, appends
    ``RETURNING id`` if not already present and reads from the cursor.
    """
    param_tuple = tuple(params)
    if is_sqlite(conn):
        cursor = execute(conn, query, param_tuple)
        return int(cursor.lastrowid)
    # PostgreSQL — ensure RETURNING clause
    sql = convert_placeholders(query, conn)
    if "RETURNING" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " RETURNING id"
    cur = conn.cursor()
    cur.execute(sql, param_tuple)
    row = cur.fetchone()
    return int(row[0]) if row else 0


def upsert_rows(
    conn: Any,
    table: str,
    pk: str,
    rows: Iterable[Dict[str, Any]],
) -> None:
    """Dialect-aware UPSERT (INSERT … ON CONFLICT … DO UPDATE).

    Args:
        conn: DB-API connection.
        table: Target table name (validated).
        pk: Primary key column name (validated).
        rows: Iterable of dicts — all dicts must have the same keys.
    """
    rows_list = list(rows)
    if not rows_list:
        return

    safe_table = safe_identifier(table)
    safe_pk = safe_identifier(pk)
    columns = list(rows_list[0].keys())
    for key in columns:
        safe_identifier(key)
    update_columns = [col for col in columns if col != safe_pk]

    if is_sqlite(conn):
        placeholders = ", ".join(["?"] * len(columns))
        assignments = ", ".join(f"{col}=excluded.{col}" for col in update_columns)
    else:
        placeholders = ", ".join(["%s"] * len(columns))
        assignments = ", ".join(f"{col}=EXCLUDED.{col}" for col in update_columns)

    conflict_sql = (
        f"ON CONFLICT({safe_pk}) DO UPDATE SET {assignments}"
        if assignments
        else f"ON CONFLICT({safe_pk}) DO NOTHING"
    )
    sql = (
        f"INSERT INTO {safe_table} ({', '.join(columns)}) "
        f"VALUES ({placeholders}) {conflict_sql}"
    )

    for row in rows_list:
        values = tuple(row.get(col) for col in columns)
        if is_sqlite(conn):
            conn.execute(sql, values)
        else:
            cur = conn.cursor()
            cur.execute(sql, values)
    commit(conn)
