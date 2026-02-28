from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import PruneCriteria
from lsm.vectordb.sqlite_vec import SQLiteVecProvider


def _vector(x: float, y: float = 0.0) -> list[float]:
    values = [0.0] * 384
    values[0] = float(x)
    values[1] = float(y)
    return values


def _provider(tmp_path: Path) -> SQLiteVecProvider:
    cfg = VectorDBConfig(provider="sqlite", path=tmp_path / "data", collection="prune")
    return SQLiteVecProvider(cfg)


def test_prune_removes_only_non_current_chunks(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["old-1", "current-1"],
        documents=["old", "current"],
        metadatas=[
            {"source_path": "/docs/a.md", "ext": ".md", "version": 1, "is_current": False},
            {"source_path": "/docs/a.md", "ext": ".md", "version": 2, "is_current": True},
        ],
        embeddings=[_vector(1.0), _vector(0.8)],
    )

    pruned = provider.prune_old_versions(PruneCriteria())
    assert pruned == 1

    rows = provider.connection.execute(
        "SELECT chunk_id, is_current FROM lsm_chunks ORDER BY chunk_id"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "current-1"
    assert int(rows[0]["is_current"]) == 1


def test_prune_respects_max_versions(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["v1", "v2", "v3", "v4"],
        documents=["d1", "d2", "d3", "d4"],
        metadatas=[
            {"source_path": "/docs/versioned.md", "ext": ".md", "version": 1, "is_current": False},
            {"source_path": "/docs/versioned.md", "ext": ".md", "version": 2, "is_current": False},
            {"source_path": "/docs/versioned.md", "ext": ".md", "version": 3, "is_current": False},
            {"source_path": "/docs/versioned.md", "ext": ".md", "version": 4, "is_current": True},
        ],
        embeddings=[_vector(0.1), _vector(0.2), _vector(0.3), _vector(0.4)],
    )

    pruned = provider.prune_old_versions(PruneCriteria(max_versions=2))
    assert pruned == 1

    versions = provider.connection.execute(
        """
        SELECT version, is_current
        FROM lsm_chunks
        WHERE source_path = ?
        ORDER BY version
        """,
        ("/docs/versioned.md",),
    ).fetchall()
    remaining = {(int(row["version"]), int(row["is_current"])) for row in versions}
    assert (1, 0) not in remaining
    assert (2, 0) in remaining
    assert (3, 0) in remaining
    assert (4, 1) in remaining


def test_prune_respects_older_than_days(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    new_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    provider.add_chunks(
        ids=["old", "new", "active"],
        documents=["old", "new", "active"],
        metadatas=[
            {
                "source_path": "/docs/age.md",
                "ext": ".md",
                "version": 1,
                "is_current": False,
                "ingested_at": old_ts,
            },
            {
                "source_path": "/docs/age.md",
                "ext": ".md",
                "version": 2,
                "is_current": False,
                "ingested_at": new_ts,
            },
            {
                "source_path": "/docs/age.md",
                "ext": ".md",
                "version": 3,
                "is_current": True,
                "ingested_at": new_ts,
            },
        ],
        embeddings=[_vector(0.1), _vector(0.2), _vector(0.3)],
    )

    pruned = provider.prune_old_versions(PruneCriteria(older_than_days=7))
    assert pruned == 1

    ids = {
        str(row["chunk_id"])
        for row in provider.connection.execute("SELECT chunk_id FROM lsm_chunks")
    }
    assert ids == {"new", "active"}


def test_prune_returns_deleted_count(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.add_chunks(
        ids=["a", "b", "c"],
        documents=["a", "b", "c"],
        metadatas=[
            {"source_path": "/docs/x.md", "ext": ".md", "version": 1, "is_current": False},
            {"source_path": "/docs/x.md", "ext": ".md", "version": 2, "is_current": False},
            {"source_path": "/docs/x.md", "ext": ".md", "version": 3, "is_current": True},
        ],
        embeddings=[_vector(0.1), _vector(0.2), _vector(0.3)],
    )

    deleted = provider.prune_old_versions(PruneCriteria())
    assert deleted == 2

