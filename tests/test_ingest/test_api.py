from __future__ import annotations

from lsm.ingest.api import (
    CollectionInfo,
    CollectionStats,
    IngestResult,
    get_collection_info,
    get_collection_stats,
    run_ingest,
    wipe_collection,
)


class FakeVectorDBProvider:
    def __init__(
        self,
        *,
        name: str = "chromadb",
        chunk_count: int = 0,
        stats: dict | None = None,
        deleted_count: int = 0,
    ) -> None:
        self.name = name
        self._chunk_count = chunk_count
        self._stats = stats or {"provider": name}
        self._deleted_count = deleted_count
        self.delete_all_calls = 0

    def count(self) -> int:
        return self._chunk_count

    def get_stats(self) -> dict:
        return self._stats

    def delete_all(self) -> int:
        self.delete_all_calls += 1
        return self._deleted_count


class ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int | None]] = []

    def __call__(self, current: int, total: int | None) -> None:
        self.calls.append((current, total))


def test_get_collection_info_uses_provider_stats(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(
        name="postgresql",
        chunk_count=7,
        stats={"provider": "postgresql"},
    )
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    info = get_collection_info(ingest_config)

    assert isinstance(info, CollectionInfo)
    assert info.provider == "postgresql"
    assert info.chunk_count == 7


def test_get_collection_info_chromadb_path(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(
        name="chromadb",
        chunk_count=3,
        stats={"name": "kb", "count": 3, "provider": "chromadb"},
    )
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    info = get_collection_info(ingest_config)

    assert info == CollectionInfo(name="kb", chunk_count=3, provider="chromadb")


def test_get_collection_stats_reports_progress(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(name="chromadb", chunk_count=10)
    progress_recorder = ProgressRecorder()
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    def fake_get_collection_stats(*_args, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            callback(4)
            callback(10)
        return {
            "total_chunks": 10,
            "unique_files": 2,
            "file_types": {".txt": 8, ".md": 2},
            "top_files": {"a.txt": 7, "b.md": 3},
        }

    monkeypatch.setattr("lsm.ingest.api._get_collection_stats", fake_get_collection_stats)

    stats = get_collection_stats(ingest_config, progress_callback=progress_recorder)

    assert stats.chunk_count == 10
    assert stats.unique_files == 2
    assert stats.file_types == {".txt": 8, ".md": 2}
    assert {"source_path": "a.txt", "chunk_count": 7} in stats.top_files
    assert {"source_path": "b.md", "chunk_count": 3} in stats.top_files
    assert (4, 10) in progress_recorder.calls
    assert (10, 10) in progress_recorder.calls


def test_get_collection_stats_fallback(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(name="postgresql", chunk_count=11)
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    def fake_get_collection_stats(*_args, **_kwargs):
        return {
            "total_chunks": 11,
            "unique_files": 0,
            "file_types": {},
            "top_files": {},
        }

    monkeypatch.setattr("lsm.ingest.api._get_collection_stats", fake_get_collection_stats)

    stats = get_collection_stats(ingest_config)

    assert isinstance(stats, CollectionStats)
    assert stats.chunk_count == 11
    assert stats.unique_files == 0
    assert stats.file_types == {}
    assert stats.top_files == []


def test_run_ingest_maps_result_and_force_resets_manifest(ingest_config, monkeypatch) -> None:
    import lsm.ingest.pipeline as ingest_pipeline

    captured_kwargs = {}

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "total_files": 5,
            "completed_files": 3,
            "skipped_files": 2,
            "chunks_added": 12,
            "elapsed_seconds": 1.25,
            "errors": [{"file": "bad.pdf", "error": "parse failed"}],
        }

    monkeypatch.setattr(ingest_pipeline, "ingest", fake_ingest)

    progress_events = []
    progress_callback = lambda event, current, total, message: progress_events.append(  # noqa: E731
        (event, current, total, message)
    )
    result = run_ingest(ingest_config, force=True, progress_callback=progress_callback)

    assert isinstance(result, IngestResult)
    assert result.total_files == 5
    assert result.completed_files == 3
    assert result.skipped_files == 2
    assert result.chunks_added == 12
    assert result.elapsed_seconds == 1.25
    assert result.errors == [{"file": "bad.pdf", "error": "parse failed"}]
    assert captured_kwargs["progress_callback"] is progress_callback
    assert captured_kwargs["force_reingest"] is True
    assert captured_kwargs["manifest_path"] is None
    assert captured_kwargs["chroma_flush_interval"] is None


def test_wipe_collection_deletes_all(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(deleted_count=3)
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    deleted = wipe_collection(ingest_config)

    assert deleted == 3
    assert provider.delete_all_calls == 1


def test_wipe_collection_empty(ingest_config, monkeypatch) -> None:
    provider = FakeVectorDBProvider(deleted_count=0)
    monkeypatch.setattr("lsm.ingest.api.create_vectordb_provider", lambda _cfg: provider)

    deleted = wipe_collection(ingest_config)

    assert deleted == 0
    assert provider.delete_all_calls == 1
