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
    RootConfig,
    VectorDBConfig,
)
from lsm.db.completion import detect_completion_mode, get_stale_files
from lsm.db.schema_version import record_schema_version
from lsm.ingest.api import run_ingest
from lsm.ingest.utils import canonical_path
from lsm.vectordb.sqlite_vec import SQLiteVecProvider


class _FakeEmbeddings:
    def __init__(self, count: int) -> None:
        row = [0.0] * 384
        row[0] = 0.1
        row[1] = 0.2
        row[2] = 0.3
        self._rows = [list(row) for _ in range(count)]

    def tolist(self) -> list[list[float]]:
        return self._rows


class _CountingSentenceTransformer:
    encode_calls = 0

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, **_kwargs):
        _CountingSentenceTransformer.encode_calls += 1
        return _FakeEmbeddings(len(texts))

    def get_sentence_embedding_dimension(self) -> int:
        return 384


def _schema_runtime(*, chunk_size: int = 1800, chunk_overlap: int = 200) -> dict[str, object]:
    return {
        "lsm_version": "0.8.0",
        "embedding_model": "fake-model",
        "embedding_dim": 384,
        "chunking_strategy": "structure",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _ensure_manifest_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lsm_manifest (
            source_path TEXT PRIMARY KEY,
            mtime_ns INTEGER,
            file_size INTEGER,
            file_hash TEXT,
            version INTEGER,
            embedding_model TEXT,
            schema_version_id INTEGER,
            updated_at TEXT
        )
        """
    )
    conn.commit()


def _insert_manifest_rows(conn: sqlite3.Connection, source_paths: list[str]) -> None:
    _ensure_manifest_table(conn)
    for path in source_paths:
        conn.execute(
            """
            INSERT OR REPLACE INTO lsm_manifest (
                source_path, mtime_ns, file_size, file_hash, version, embedding_model, schema_version_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, 1, 1, "hash", 1, "fake-model", 1, "2026-01-01T00:00:00+00:00"),
        )
    conn.commit()


def _build_config(
    root: Path,
    tmp_path: Path,
    *,
    chunk_size: int = 200,
    root_tags: list[str] | None = None,
    content_type: str | None = None,
) -> LSMConfig:
    if root_tags or content_type:
        roots = [
            RootConfig(path=root, tags=root_tags, content_type=content_type),
        ]
    else:
        roots = [RootConfig(path=root)]
    return LSMConfig(
        ingest=IngestConfig(
            roots=roots,
            extensions=[".txt", ".md", ".pdf"],
            override_extensions=True,
            exclude_dirs=[],
            override_excludes=True,
            skip_errors=True,
            chunk_size=chunk_size,
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
            collection="completion_test",
        ),
        global_settings=GlobalConfig(
            global_folder=tmp_path / "global",
            embed_model="fake-model",
            device="cpu",
            batch_size=8,
        ),
    )


def test_new_extension_completion_reingests_only_new_extension_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir(parents=True, exist_ok=True)
    md_file = root / "a.md"
    pdf_file = root / "b.pdf"
    md_file.write_text("alpha", encoding="utf-8")
    pdf_file.write_text("beta", encoding="utf-8")

    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_runtime())
    _insert_manifest_rows(conn, [canonical_path(md_file)])

    config = {
        **_schema_runtime(),
        "roots": [RootConfig(path=root)],
        "exts": {".md", ".pdf"},
        "exclude_dirs": set(),
    }
    mode = detect_completion_mode(conn, config)
    assert mode == "extension_completion"

    stale = get_stale_files(conn, config, mode)
    assert stale == [canonical_path(pdf_file)]


def test_chunk_size_change_reingests_all_manifest_files(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_runtime(chunk_size=1800))
    paths = ["/docs/a.md", "/docs/b.md"]
    _insert_manifest_rows(conn, paths)

    config = {
        **_schema_runtime(chunk_size=1200),
        "roots": [RootConfig(path=tmp_path / "docs")],
        "exts": {".md"},
        "exclude_dirs": set(),
    }
    mode = detect_completion_mode(conn, config)
    assert mode == "chunk_boundary_update"
    assert get_stale_files(conn, config, mode) == sorted(paths)


def test_unchanged_config_reports_no_completion(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir(parents=True, exist_ok=True)
    md_file = root / "a.md"
    md_file.write_text("alpha", encoding="utf-8")

    conn = sqlite3.connect(str(tmp_path / "lsm.db"))
    record_schema_version(conn, _schema_runtime())
    _insert_manifest_rows(conn, [canonical_path(md_file)])

    config = {
        **_schema_runtime(),
        "roots": [RootConfig(path=root)],
        "exts": {".md"},
        "exclude_dirs": set(),
    }
    assert detect_completion_mode(conn, config) is None


def test_metadata_enrichment_updates_without_reembedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "a.txt").write_text("alpha " * 200, encoding="utf-8")

    config_initial = _build_config(docs_root, tmp_path, root_tags=None, content_type=None)
    provider = SQLiteVecProvider(config_initial.vectordb)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_CountingSentenceTransformer),
    )
    monkeypatch.setattr(
        "lsm.ingest.pipeline.create_vectordb_provider",
        lambda *_args, **_kwargs: provider,
    )

    _CountingSentenceTransformer.encode_calls = 0
    first = run_ingest(config_initial, force=True)
    assert first.chunks_added > 0
    first_encode_calls = _CountingSentenceTransformer.encode_calls
    assert first_encode_calls > 0

    config_enriched = _build_config(
        docs_root,
        tmp_path,
        root_tags=["theory"],
        content_type="notes",
    )
    second = run_ingest(
        config_enriched,
        force=False,
        force_reingest_changed_config=True,
    )
    assert second.chunks_added == 0
    assert _CountingSentenceTransformer.encode_calls == first_encode_calls

    row = provider.connection.execute(
        """
        SELECT root_tags, content_type
        FROM lsm_chunks
        WHERE source_path = ? AND is_current = 1
        LIMIT 1
        """,
        (canonical_path(docs_root / "a.txt"),),
    ).fetchone()
    assert row is not None
    assert "theory" in str(row["root_tags"] or "")
    assert row["content_type"] == "notes"


def test_injected_write_failure_rolls_back_chunks_and_manifest_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    (docs_root / "a.txt").write_text("alpha " * 200, encoding="utf-8")

    config = _build_config(docs_root, tmp_path)
    provider = SQLiteVecProvider(config.vectordb)
    original_add_chunks = provider.add_chunks

    def _failing_add_chunks(ids, documents, metadatas, embeddings):
        original_add_chunks(ids, documents, metadatas, embeddings)
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(provider, "add_chunks", _failing_add_chunks)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_CountingSentenceTransformer),
    )
    monkeypatch.setattr(
        "lsm.ingest.pipeline.create_vectordb_provider",
        lambda *_args, **_kwargs: provider,
    )

    with pytest.raises(RuntimeError, match="write stage failed"):
        run_ingest(config, force=True)

    chunks_count = provider.connection.execute(
        "SELECT COUNT(*) FROM lsm_chunks"
    ).fetchone()[0]
    manifest_count = provider.connection.execute(
        "SELECT COUNT(*) FROM lsm_manifest"
    ).fetchone()[0]
    assert int(chunks_count) == 0
    assert int(manifest_count) == 0
