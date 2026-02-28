from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.config.models import VectorDBConfig
from lsm.db.connection import (
    create_sqlite_connection,
    resolve_db_path,
    resolve_postgres_connection_factory,
    resolve_sqlite_connection,
    resolve_vectordb_provider_name,
)


def test_create_sqlite_connection_sets_wal_and_fk(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = create_sqlite_connection(db_path)
    try:
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert journal == "wal"
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000
        # Verify row_factory is set
        row = conn.execute("SELECT 1 AS value").fetchone()
        assert row["value"] == 1
    finally:
        conn.close()


def test_resolve_db_path_appends_lsm_db_for_directory(tmp_path: Path) -> None:
    result = resolve_db_path(tmp_path)
    assert result == tmp_path / "lsm.db"


def test_resolve_db_path_returns_unchanged_for_db_extension(tmp_path: Path) -> None:
    db_file = tmp_path / "custom.db"
    result = resolve_db_path(db_file)
    assert result == db_file


def test_resolve_sqlite_connection_from_provider_instance(tmp_path: Path) -> None:
    inner_conn = sqlite3.connect(str(tmp_path / "test.db"))
    try:
        provider = SimpleNamespace(
            name="sqlite",
            config=VectorDBConfig(provider="sqlite", path=tmp_path),
            connection=inner_conn,
        )
        conn, owns = resolve_sqlite_connection(provider)
        assert conn is inner_conn
        assert owns is False
    finally:
        inner_conn.close()


def test_resolve_sqlite_connection_creates_provider_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner_conn = sqlite3.connect(str(tmp_path / "test.db"))

    class _FakeProvider:
        name = "sqlite"
        config = VectorDBConfig(provider="sqlite", path=tmp_path)
        connection = inner_conn

    monkeypatch.setattr(
        "lsm.db.connection.create_vectordb_provider",
        lambda cfg: _FakeProvider(),
        raising=False,
    )
    # Patch the import path used inside resolve_sqlite_connection
    import lsm.vectordb as vectordb_mod
    monkeypatch.setattr(vectordb_mod, "create_vectordb_provider", lambda cfg: _FakeProvider())

    config = VectorDBConfig(provider="sqlite", path=tmp_path)
    conn, owns = resolve_sqlite_connection(config)
    assert conn is inner_conn
    assert owns is True
    inner_conn.close()


def test_resolve_postgres_connection_factory_returns_callable() -> None:
    factory = lambda: None  # noqa: E731
    provider = SimpleNamespace(
        name="postgresql",
        config=SimpleNamespace(provider="postgresql"),
        _get_conn=factory,
    )
    result = resolve_postgres_connection_factory(provider)
    assert result is factory


def test_resolve_postgres_connection_factory_returns_none_for_sqlite() -> None:
    provider = SimpleNamespace(
        name="sqlite",
        config=SimpleNamespace(provider="sqlite"),
    )
    result = resolve_postgres_connection_factory(provider)
    assert result is None


def test_resolve_vectordb_provider_name_from_provider() -> None:
    provider = SimpleNamespace(name="sqlite", config=SimpleNamespace(provider="sqlite"))
    assert resolve_vectordb_provider_name(provider) == "sqlite"


def test_resolve_vectordb_provider_name_from_config() -> None:
    config = VectorDBConfig(provider="postgresql", path=Path("/tmp"))
    assert resolve_vectordb_provider_name(config) == "postgresql"
