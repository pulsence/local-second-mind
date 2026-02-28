from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.config.models import (
    GlobalConfig,
    IngestConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LSMConfig,
    QueryConfig,
    VectorDBConfig,
)
from lsm.db.schema_version import (
    SchemaVersionMismatchError,
    check_schema_compatibility,
    get_active_schema_version,
    record_schema_version,
)
from lsm.ingest.api import run_ingest
from lsm.ingest.manifest import load_manifest


class _FakeEmbeddings:
    def __init__(self, count: int) -> None:
        self._rows = [[0.1, 0.2, 0.3] for _ in range(count)]

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, **_kwargs):
        return _FakeEmbeddings(len(texts))

    def get_sentence_embedding_dimension(self) -> int:
        return 384


class _InMemoryProvider:
    def __init__(self, db_path: Path) -> None:
        self.name = "sqlite"
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row
        self._rows: dict[str, dict] = {}

    def add_chunks(self, ids, documents, metadatas, embeddings) -> None:
        for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self._rows[cid] = {"doc": doc, "meta": dict(meta or {}), "emb": emb}

    def get(self, ids=None, filters=None, include=None):
        _ = ids, include
        source_path = (filters or {}).get("source_path")
        if not source_path:
            selected = list(self._rows.keys())
        else:
            selected = [
                cid for cid, row in self._rows.items()
                if row["meta"].get("source_path") == source_path
            ]
        return SimpleNamespace(ids=selected)

    def update_metadatas(self, ids, metadatas) -> None:
        for cid, metadata in zip(ids, metadatas):
            row = self._rows.get(cid)
            if not row:
                continue
            updated = dict(row["meta"])
            updated.update(metadata or {})
            row["meta"] = updated

    def delete_by_filter(self, filters) -> None:
        source_path = (filters or {}).get("source_path")
        if not source_path:
            return
        for cid in [
            key
            for key, row in self._rows.items()
            if row["meta"].get("source_path") == source_path
        ]:
            self._rows.pop(cid, None)

    def count(self) -> int:
        return len(self._rows)


def _schema_config(**overrides):
    config = {
        "lsm_version": "0.8.0",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "chunking_strategy": "structure",
        "chunk_size": 1800,
        "chunk_overlap": 200,
    }
    config.update(overrides)
    return config


def test_first_ingest_records_schema_version(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    schema_id = record_schema_version(conn, _schema_config())
    active = get_active_schema_version(conn)
    assert active is not None
    assert int(active["id"]) == schema_id
    assert active["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_matching_config_passes_compatibility_check(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_config())
    compatible, diff = check_schema_compatibility(conn, _schema_config())
    assert compatible is True
    assert diff == {}


def test_changed_embed_model_fails_with_clear_error(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_config())
    with pytest.raises(SchemaVersionMismatchError, match="lsm migrate"):
        check_schema_compatibility(
            conn,
            _schema_config(embedding_model="text-embedding-3-large"),
            raise_on_mismatch=True,
        )


def test_changed_chunk_size_fails_with_clear_error(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_config())
    with pytest.raises(SchemaVersionMismatchError, match="chunk_size"):
        check_schema_compatibility(
            conn,
            _schema_config(chunk_size=1024),
            raise_on_mismatch=True,
        )


def test_manifest_entries_reference_current_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "a.txt").write_text("alpha " * 400, encoding="utf-8")
    provider = _InMemoryProvider(tmp_path / "lsm.db")

    config = LSMConfig(
        ingest=IngestConfig(
            roots=[docs_root],
            extensions=[".txt"],
            override_extensions=True,
            exclude_dirs=[],
            override_excludes=True,
            skip_errors=True,
            chunk_size=200,
            chunk_overlap=40,
            chunking_strategy="structure",
        ),
        query=QueryConfig(),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="local")],
            services={"default": LLMServiceConfig(provider="local", model="tiny")},
        ),
        vectordb=VectorDBConfig(
            provider="sqlite",
            path=tmp_path / "data",
            collection="schema_version_test",
        ),
        global_settings=GlobalConfig(
            global_folder=tmp_path / "global",
            embed_model="fake-model",
            device="cpu",
            batch_size=8,
        ),
    )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(
        "lsm.ingest.pipeline.create_vectordb_provider",
        lambda *_args, **_kwargs: provider,
    )

    run_ingest(config, force=True)

    row = provider.connection.execute(
        "SELECT id FROM lsm_schema_versions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    schema_version_id = int(row[0])

    manifest = load_manifest(connection=provider.connection)
    assert manifest
    for entry in manifest.values():
        assert int(entry["schema_version_id"]) == schema_version_id
        assert entry.get("embedding_model") == "fake-model"

