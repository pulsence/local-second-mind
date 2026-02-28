from __future__ import annotations

import sqlite3

import pytest

from lsm.db.transaction import transaction


def _test_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.commit()
    return conn


def test_transaction_commits_on_success() -> None:
    conn = _test_conn()
    with transaction(conn):
        conn.execute("INSERT INTO t(id, val) VALUES (1, 'a')")

    row = conn.execute("SELECT val FROM t WHERE id = 1").fetchone()
    assert row is not None
    assert row["val"] == "a"
    conn.close()


def test_transaction_rolls_back_on_exception() -> None:
    conn = _test_conn()
    with pytest.raises(RuntimeError):
        with transaction(conn):
            conn.execute("INSERT INTO t(id, val) VALUES (1, 'a')")
            raise RuntimeError("boom")

    row = conn.execute("SELECT val FROM t WHERE id = 1").fetchone()
    assert row is None
    conn.close()


def test_nested_transactions_use_savepoints() -> None:
    conn = _test_conn()
    with transaction(conn):
        conn.execute("INSERT INTO t(id, val) VALUES (1, 'outer')")
        with transaction(conn):
            conn.execute("INSERT INTO t(id, val) VALUES (2, 'inner')")

    rows = conn.execute("SELECT id FROM t ORDER BY id").fetchall()
    assert [row["id"] for row in rows] == [1, 2]
    conn.close()


def test_nested_rollback_does_not_affect_outer() -> None:
    conn = _test_conn()
    with transaction(conn):
        conn.execute("INSERT INTO t(id, val) VALUES (1, 'outer')")
        with pytest.raises(RuntimeError):
            with transaction(conn):
                conn.execute("INSERT INTO t(id, val) VALUES (2, 'inner')")
                raise RuntimeError("inner boom")
        # outer should still be intact
        conn.execute("INSERT INTO t(id, val) VALUES (3, 'after')")

    rows = conn.execute("SELECT id FROM t ORDER BY id").fetchall()
    ids = [row["id"] for row in rows]
    assert 1 in ids
    assert 2 not in ids
    assert 3 in ids
    conn.close()
