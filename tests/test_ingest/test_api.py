from lsm.ingest.api import (
    CollectionInfo,
    CollectionStats,
    IngestResult,
    get_collection_info,
    get_collection_stats,
    run_ingest,
    wipe_collection,
)


def test_get_collection_info_uses_provider_stats(
    ingest_config,
    mock_vectordb_provider,
    mocker,
) -> None:
    mock_vectordb_provider.name = "postgresql"
    mock_vectordb_provider.count.return_value = 7
    mock_vectordb_provider.get_stats.return_value = {"provider": "postgresql"}
    mocker.patch(
        "lsm.ingest.api.create_vectordb_provider",
        return_value=mock_vectordb_provider,
    )

    info = get_collection_info(ingest_config)

    assert isinstance(info, CollectionInfo)
    assert info.provider == "postgresql"
    assert info.chunk_count == 7


def test_get_collection_info_chromadb_path(ingest_config, mock_vectordb_provider, mocker) -> None:
    mock_vectordb_provider.name = "chromadb"
    mock_vectordb_provider.count.return_value = 3
    mock_vectordb_provider.get_stats.return_value = {"name": "kb", "count": 3, "provider": "chromadb"}
    mocker.patch(
        "lsm.ingest.api.create_vectordb_provider",
        return_value=mock_vectordb_provider,
    )

    info = get_collection_info(ingest_config)

    assert info == CollectionInfo(name="kb", chunk_count=3, provider="chromadb")


def test_get_collection_stats_reports_progress(
    ingest_config,
    mock_vectordb_provider,
    progress_callback_mock,
    mocker,
) -> None:
    mock_vectordb_provider.name = "chromadb"
    mock_vectordb_provider.count.return_value = 10
    mocker.patch(
        "lsm.ingest.api.create_vectordb_provider",
        return_value=mock_vectordb_provider,
    )

    def fake_get_collection_stats(*args, **kwargs):
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

    mocker.patch("lsm.ingest.api._get_collection_stats", side_effect=fake_get_collection_stats)

    stats = get_collection_stats(ingest_config, progress_callback=progress_callback_mock)

    assert stats.chunk_count == 10
    assert stats.unique_files == 2
    assert stats.file_types == {".txt": 8, ".md": 2}
    assert {"source_path": "a.txt", "chunk_count": 7} in stats.top_files
    assert {"source_path": "b.md", "chunk_count": 3} in stats.top_files
    progress_callback_mock.assert_any_call(4, 10)
    progress_callback_mock.assert_any_call(10, 10)


def test_get_collection_stats_fallback(
    ingest_config,
    mock_vectordb_provider,
    mocker,
) -> None:
    mock_vectordb_provider.name = "postgresql"
    mock_vectordb_provider.count.return_value = 11
    mocker.patch(
        "lsm.ingest.api.create_vectordb_provider",
        return_value=mock_vectordb_provider,
    )

    def fake_get_collection_stats(*args, **kwargs):
        return {
            "total_chunks": 11,
            "unique_files": 0,
            "file_types": {},
            "top_files": {},
        }

    mocker.patch("lsm.ingest.api._get_collection_stats", side_effect=fake_get_collection_stats)

    stats = get_collection_stats(ingest_config)

    assert isinstance(stats, CollectionStats)
    assert stats.chunk_count == 11
    assert stats.unique_files == 0
    assert stats.file_types == {}
    assert stats.top_files == []


def test_run_ingest_maps_result_and_force_resets_manifest(
    ingest_config,
    progress_callback_mock,
    mocker,
) -> None:
    ingest_config.ingest.manifest.parent.mkdir(parents=True, exist_ok=True)
    ingest_config.ingest.manifest.write_text("{}", encoding="utf-8")
    ingest_mock = mocker.patch(
        "lsm.ingest.pipeline.ingest",
        return_value={
            "total_files": 5,
            "completed_files": 3,
            "skipped_files": 2,
            "chunks_added": 12,
            "elapsed_seconds": 1.25,
            "errors": [{"file": "bad.pdf", "error": "parse failed"}],
        },
    )

    result = run_ingest(ingest_config, force=True, progress_callback=progress_callback_mock)

    assert isinstance(result, IngestResult)
    assert result.total_files == 5
    assert result.completed_files == 3
    assert result.skipped_files == 2
    assert result.chunks_added == 12
    assert result.elapsed_seconds == 1.25
    assert result.errors == [{"file": "bad.pdf", "error": "parse failed"}]
    assert not ingest_config.ingest.manifest.exists()
    assert ingest_mock.call_args.kwargs["progress_callback"] is progress_callback_mock


def test_wipe_collection_deletes_all(ingest_config, mocker) -> None:
    provider = mocker.MagicMock()
    provider.delete_all.return_value = 3
    mocker.patch("lsm.ingest.api.create_vectordb_provider", return_value=provider)

    deleted = wipe_collection(ingest_config)

    assert deleted == 3
    provider.delete_all.assert_called_once()


def test_wipe_collection_empty(ingest_config, mocker) -> None:
    provider = mocker.MagicMock()
    provider.delete_all.return_value = 0
    mocker.patch("lsm.ingest.api.create_vectordb_provider", return_value=provider)

    deleted = wipe_collection(ingest_config)

    assert deleted == 0
    provider.delete_all.assert_called_once()
