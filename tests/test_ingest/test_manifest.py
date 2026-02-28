from __future__ import annotations

import sqlite3

from lsm.ingest.manifest import get_next_version, load_manifest, save_manifest


def test_load_manifest_missing_file_returns_empty_dict(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert load_manifest(missing) == {}


def test_load_manifest_invalid_json_returns_empty_dict(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid json", encoding="utf-8")
    assert load_manifest(manifest_path) == {}


def test_manifest_roundtrip_uses_db_connection(tmp_path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    payload = {
        "/docs/a.txt": {
            "mtime_ns": 100,
            "size": 12,
            "file_hash": "abc",
            "version": 1,
            "updated_at": "2026-02-28T12:00:00+00:00",
        }
    }

    save_manifest(None, payload, connection=conn)
    loaded = load_manifest(connection=conn)

    assert loaded == payload


def test_manifest_save_prunes_removed_rows(tmp_path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    save_manifest(
        None,
        {
            "/docs/a.txt": {"mtime_ns": 1, "size": 1, "file_hash": "a", "version": 1},
            "/docs/b.txt": {"mtime_ns": 2, "size": 2, "file_hash": "b", "version": 1},
        },
        connection=conn,
    )
    save_manifest(
        None,
        {"/docs/a.txt": {"mtime_ns": 10, "size": 10, "file_hash": "a2", "version": 2}},
        connection=conn,
    )

    loaded = load_manifest(connection=conn)
    assert set(loaded.keys()) == {"/docs/a.txt"}
    assert loaded["/docs/a.txt"]["version"] == 2


def test_get_next_version_queries_database(tmp_path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    save_manifest(
        None,
        {"/docs/a.txt": {"mtime_ns": 1, "size": 1, "file_hash": "x", "version": 3}},
        connection=conn,
    )

    assert get_next_version({}, "/docs/a.txt", connection=conn) == 4
    assert get_next_version({}, "/docs/new.txt", connection=conn) == 1
