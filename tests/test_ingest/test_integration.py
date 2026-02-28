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
from lsm.ingest.api import run_ingest
from lsm.ingest.manifest import load_manifest


class InMemoryVectorProvider:
    def __init__(self, db_path: Path) -> None:
        self._rows = {}
        self.name = "sqlite"
        self.connection = sqlite3.connect(str(db_path))
        self.connection.row_factory = sqlite3.Row

    def add_chunks(self, ids, documents, metadatas, embeddings) -> None:
        for cid, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
            self._rows[cid] = {
                "doc": doc,
                "meta": meta,
                "emb": emb,
            }

    def delete_by_filter(self, filters) -> None:
        source_path = (filters or {}).get("source_path")
        if not source_path:
            return
        for cid in [k for k, v in self._rows.items() if v["meta"].get("source_path") == source_path]:
            self._rows.pop(cid, None)

    def get(self, ids=None, filters=None, include=None):
        _ = ids, include
        source_path = (filters or {}).get("source_path")
        if not source_path:
            selected = list(self._rows.keys())
        else:
            selected = [
                cid
                for cid, row in self._rows.items()
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

    def count(self) -> int:
        return len(self._rows)


class FakeEmbeddings:
    def __init__(self, count: int) -> None:
        self._data = [[0.1, 0.2, 0.3] for _ in range(count)]

    def tolist(self):
        return self._data


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def encode(self, texts, **_kwargs):
        return FakeEmbeddings(len(texts))

    def get_sentence_embedding_dimension(self) -> int:
        return 384


def _build_config(root: Path, tmp_path: Path) -> LSMConfig:
    return LSMConfig(
        ingest=IngestConfig(
            roots=[root],
            extensions=[".txt", ".md", ".html"],
            override_extensions=True,
            exclude_dirs=[],
            override_excludes=True,
            skip_errors=True,
        ),
        query=QueryConfig(),
        llm=LLMRegistryConfig(
            providers=[LLMProviderConfig(provider_name="local")],
            services={"query": LLMServiceConfig(provider="local", model="llama3.1")}
        ),
        vectordb=VectorDBConfig(
            provider="sqlite",
            path=tmp_path / "data",
            collection="integration_test_collection",
        ),
        global_settings=GlobalConfig(
            global_folder=tmp_path / "global",
            embed_model="fake-model",
            device="cpu",
            batch_size=8,
        ),
    )


@pytest.mark.integration
class TestIngestIntegration:
    def test_ingest_empty_directory(self, tmp_path: Path, monkeypatch) -> None:
        provider = InMemoryVectorProvider(tmp_path / "lsm.db")
        empty_root = tmp_path / "empty_docs"
        empty_root.mkdir()
        config = _build_config(empty_root, tmp_path)
        progress_events = []

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        )
        monkeypatch.setattr(
            "lsm.ingest.pipeline.create_vectordb_provider",
            lambda *_args, **_kwargs: provider,
        )

        result = run_ingest(
            config,
            force=True,
            progress_callback=lambda event, current, total, message: progress_events.append(
                (event, current, total, message)
            ),
        )

        assert result.total_files == 0
        assert result.completed_files == 0
        assert result.skipped_files == 0
        assert result.chunks_added == 0
        assert any(event[0] == "discovery" for event in progress_events)
        assert any(event[0] == "complete" for event in progress_events)

    def test_ingest_with_text_files(self, synthetic_data_root: Path, tmp_path: Path, monkeypatch) -> None:
        provider = InMemoryVectorProvider(tmp_path / "lsm.db")
        config = _build_config(synthetic_data_root, tmp_path)

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        )
        monkeypatch.setattr(
            "lsm.ingest.pipeline.create_vectordb_provider",
            lambda *_args, **_kwargs: provider,
        )

        result = run_ingest(config, force=True)

        assert result.total_files >= 4
        assert result.completed_files == result.total_files
        assert result.skipped_files >= 1
        assert any(error.get("error") == "empty_text" for error in result.errors)
        assert result.chunks_added > 0
        assert provider.count() == result.chunks_added

        manifest_data = load_manifest(connection=provider.connection)
        assert len(manifest_data) == result.total_files - result.skipped_files
        assert not (tmp_path / ".ingest" / "manifest.json").exists()

    def test_incremental_ingest_skips_unchanged(
        self,
        synthetic_data_root: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        provider = InMemoryVectorProvider(tmp_path / "lsm.db")
        config = _build_config(synthetic_data_root, tmp_path)

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
        )
        monkeypatch.setattr(
            "lsm.ingest.pipeline.create_vectordb_provider",
            lambda *_args, **_kwargs: provider,
        )

        first = run_ingest(config, force=True)
        second = run_ingest(config, force=False)

        assert first.total_files >= 4
        assert second.total_files == first.total_files
        assert second.skipped_files == second.total_files
        assert second.chunks_added == 0
