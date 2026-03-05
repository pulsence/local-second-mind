"""Tests for lsm.db.compat — cross-backend SQL compatibility layer."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from lsm.db.compat import (
    DBOperationalError,
    commit,
    convert_placeholders,
    count_rows,
    db_error,
    dialect,
    execute,
    execute_ddl_script,
    executemany,
    fetch_rows_as_dicts,
    fetchall,
    fetchone,
    insert_returning_id,
    is_postgres,
    is_sqlite,
    row_to_dict,
    safe_identifier,
    table_exists,
    upsert_rows,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    """Return a fresh in-memory SQLite connection with Row factory."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class _FakePgCursor:
    """Minimal cursor stub that mimics psycopg2 cursor behaviour."""

    def __init__(self) -> None:
        self.last_query: str | None = None
        self.last_params: tuple = ()
        self._rows: list[tuple] = []
        self.description: list[tuple] | None = None
        self.lastrowid = 0

    def execute(self, query: str, params: tuple = ()) -> None:
        self.last_query = query
        self.last_params = params

    def executemany(self, query: str, params_seq: list[tuple]) -> None:
        for p in params_seq:
            self.execute(query, p)

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple]:
        return list(self._rows)


class _FakePgConn:
    """Minimal connection stub that pretends to be a psycopg2 connection."""

    def __init__(self) -> None:
        self._cursor = _FakePgCursor()
        self.committed = False

    def cursor(self) -> _FakePgCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


# ------------------------------------------------------------------
# Dialect detection
# ------------------------------------------------------------------

def test_dialect_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    assert dialect(sqlite_conn) == "sqlite"
    assert is_sqlite(sqlite_conn) is True
    assert is_postgres(sqlite_conn) is False


def test_dialect_postgres() -> None:
    pg = _FakePgConn()
    assert dialect(pg) == "postgresql"
    assert is_sqlite(pg) is False
    assert is_postgres(pg) is True


# ------------------------------------------------------------------
# Placeholder conversion
# ------------------------------------------------------------------

def test_convert_placeholders_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    assert convert_placeholders("SELECT ? WHERE x = ?", sqlite_conn) == "SELECT ? WHERE x = ?"


def test_convert_placeholders_postgres() -> None:
    pg = _FakePgConn()
    assert convert_placeholders("SELECT ? WHERE x = ?", pg) == "SELECT %s WHERE x = %s"


# ------------------------------------------------------------------
# safe_identifier
# ------------------------------------------------------------------

def test_safe_identifier_valid() -> None:
    assert safe_identifier("lsm_chunks") == "lsm_chunks"
    assert safe_identifier("_private") == "_private"
    assert safe_identifier("Table123") == "Table123"


def test_safe_identifier_rejects_injection() -> None:
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        safe_identifier("lsm_chunks; DROP TABLE")

    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        safe_identifier("Robert'); --")

    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        safe_identifier("")


# ------------------------------------------------------------------
# table_exists
# ------------------------------------------------------------------

def test_table_exists_sqlite_true(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)")
    assert table_exists(sqlite_conn, "test_tbl") is True


def test_table_exists_sqlite_false(sqlite_conn: sqlite3.Connection) -> None:
    assert table_exists(sqlite_conn, "nonexistent") is False


# ------------------------------------------------------------------
# execute / fetchone / fetchall
# ------------------------------------------------------------------

def test_execute_and_fetch_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (v TEXT)")
    execute(sqlite_conn, "INSERT INTO t VALUES (?)", ("hello",))
    row = fetchone(sqlite_conn, "SELECT v FROM t")
    assert row is not None
    assert row["v"] == "hello"

    execute(sqlite_conn, "INSERT INTO t VALUES (?)", ("world",))
    rows = fetchall(sqlite_conn, "SELECT v FROM t ORDER BY v")
    assert len(rows) == 2
    assert rows[0]["v"] == "hello"


# ------------------------------------------------------------------
# executemany
# ------------------------------------------------------------------

def test_executemany_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (v TEXT)")
    executemany(sqlite_conn, "INSERT INTO t VALUES (?)", [("a",), ("b",), ("c",)])
    rows = fetchall(sqlite_conn, "SELECT v FROM t ORDER BY v")
    assert len(rows) == 3


# ------------------------------------------------------------------
# execute_ddl_script
# ------------------------------------------------------------------

def test_execute_ddl_script_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    ddl = """
        CREATE TABLE ddl_a (id INTEGER PRIMARY KEY);
        CREATE TABLE ddl_b (id INTEGER PRIMARY KEY);
        CREATE INDEX idx_ddl_a ON ddl_a(id);
    """
    execute_ddl_script(sqlite_conn, ddl)
    assert table_exists(sqlite_conn, "ddl_a") is True
    assert table_exists(sqlite_conn, "ddl_b") is True


# ------------------------------------------------------------------
# row_to_dict
# ------------------------------------------------------------------

def test_row_to_dict_sqlite_row(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (a TEXT, b INTEGER)")
    sqlite_conn.execute("INSERT INTO t VALUES ('x', 1)")
    row = sqlite_conn.execute("SELECT a, b FROM t").fetchone()
    d = row_to_dict(row)
    assert d == {"a": "x", "b": 1}


def test_row_to_dict_tuple_with_columns() -> None:
    d = row_to_dict((1, "foo"), columns=["id", "name"])
    assert d == {"id": 1, "name": "foo"}


def test_row_to_dict_mapping() -> None:
    d = row_to_dict({"a": 1, "b": 2})
    assert d == {"a": 1, "b": 2}


def test_row_to_dict_tuple_without_columns_raises() -> None:
    with pytest.raises(TypeError, match="Cannot convert row"):
        row_to_dict((1, 2, 3))


def test_row_to_dict_none() -> None:
    assert row_to_dict(None) == {}


# ------------------------------------------------------------------
# fetch_rows_as_dicts
# ------------------------------------------------------------------

def test_fetch_rows_as_dicts_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (a TEXT, b INTEGER)")
    sqlite_conn.execute("INSERT INTO t VALUES ('x', 1)")
    sqlite_conn.execute("INSERT INTO t VALUES ('y', 2)")
    rows = fetch_rows_as_dicts(sqlite_conn, "SELECT a, b FROM t ORDER BY b")
    assert rows == [{"a": "x", "b": 1}, {"a": "y", "b": 2}]


# ------------------------------------------------------------------
# insert_returning_id
# ------------------------------------------------------------------

def test_insert_returning_id_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute(
        "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)"
    )
    row_id = insert_returning_id(
        sqlite_conn,
        "INSERT INTO t (v) VALUES (?)",
        ("hello",),
    )
    assert row_id == 1
    row_id2 = insert_returning_id(
        sqlite_conn,
        "INSERT INTO t (v) VALUES (?)",
        ("world",),
    )
    assert row_id2 == 2


# ------------------------------------------------------------------
# count_rows
# ------------------------------------------------------------------

def test_count_rows_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (v TEXT)")
    assert count_rows(sqlite_conn, "t") == 0
    sqlite_conn.execute("INSERT INTO t VALUES ('a')")
    sqlite_conn.execute("INSERT INTO t VALUES ('b')")
    assert count_rows(sqlite_conn, "t") == 2


def test_count_rows_nonexistent(sqlite_conn: sqlite3.Connection) -> None:
    assert count_rows(sqlite_conn, "nonexistent") == 0


# ------------------------------------------------------------------
# upsert_rows
# ------------------------------------------------------------------

def test_upsert_rows_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT)")
    upsert_rows(sqlite_conn, "t", "k", [
        {"k": "a", "v": "1"},
        {"k": "b", "v": "2"},
    ])
    assert count_rows(sqlite_conn, "t") == 2
    # Upsert — update existing
    upsert_rows(sqlite_conn, "t", "k", [{"k": "a", "v": "updated"}])
    row = fetchone(sqlite_conn, "SELECT v FROM t WHERE k = ?", ("a",))
    assert row["v"] == "updated"


def test_upsert_rows_empty(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (k TEXT PRIMARY KEY, v TEXT)")
    upsert_rows(sqlite_conn, "t", "k", [])
    assert count_rows(sqlite_conn, "t") == 0


# ------------------------------------------------------------------
# commit
# ------------------------------------------------------------------

def test_commit_sqlite(sqlite_conn: sqlite3.Connection) -> None:
    sqlite_conn.execute("CREATE TABLE t (v TEXT)")
    sqlite_conn.execute("INSERT INTO t VALUES ('x')")
    commit(sqlite_conn)
    # Verify data persists after commit
    row = fetchone(sqlite_conn, "SELECT v FROM t")
    assert row is not None


# ------------------------------------------------------------------
# db_error
# ------------------------------------------------------------------

def test_db_error_catches_sqlite_operational() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(DBOperationalError):
            with db_error():
                conn.execute("SELECT * FROM nonexistent_table")
    finally:
        conn.close()


def test_db_error_passes_normal_exceptions() -> None:
    """Non-DB exceptions should not be caught."""
    with pytest.raises(ValueError):
        with db_error():
            raise ValueError("not a db error")


# ------------------------------------------------------------------
# Module imports without psycopg2
# ------------------------------------------------------------------

def test_compat_imports_without_psycopg2() -> None:
    """Verify that compat module can be imported in SQLite-only environments."""
    import importlib
    import lsm.db.compat as mod
    # Re-import to ensure no psycopg2 import error
    importlib.reload(mod)
    assert hasattr(mod, "dialect")
    assert hasattr(mod, "execute")
