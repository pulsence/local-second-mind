"""Tests for ChromaDB-to-PostgreSQL migration tool."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from lsm.config.models import VectorDBConfig
from lsm.vectordb.base import VectorDBGetResult


class TestMigrateChromadbToPostgres:
    def test_rejects_non_postgres_config(self) -> None:
        from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres
        from pathlib import Path

        cfg = VectorDBConfig(provider="chromadb", path=Path("/tmp"))
        with pytest.raises(ValueError, match="must be 'postgresql'"):
            migrate_chromadb_to_postgres(Path("/tmp"), "kb", cfg)

    @patch("lsm.vectordb.migrations.chromadb_to_postgres.PostgreSQLProvider")
    @patch("lsm.vectordb.migrations.chromadb_to_postgres.ChromaDBProvider")
    def test_migrates_all_chunks(self, mock_chroma_cls, mock_pg_cls) -> None:
        from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres
        from pathlib import Path

        # Set up mock source provider
        mock_source = MagicMock()
        mock_chroma_cls.return_value = mock_source
        mock_source.count.return_value = 3
        mock_source.get.side_effect = [
            VectorDBGetResult(
                ids=["a", "b", "c"],
                documents=["doc1", "doc2", "doc3"],
                metadatas=[{"k": "1"}, {"k": "2"}, {"k": "3"}],
                embeddings=[[0.1], [0.2], [0.3]],
            ),
            VectorDBGetResult(ids=[]),
        ]

        # Set up mock target provider
        mock_target = MagicMock()
        mock_pg_cls.return_value = mock_target

        pg_config = VectorDBConfig(provider="postgresql")
        result = migrate_chromadb_to_postgres(Path("/tmp/chroma"), "kb", pg_config, batch_size=100)

        assert result["migrated"] == 3
        assert result["total"] == 3
        assert result["source"] == "chromadb"
        assert result["target"] == "postgresql"
        assert result["collection"] == "kb"

        mock_target.add_chunks.assert_called_once_with(
            ["a", "b", "c"],
            ["doc1", "doc2", "doc3"],
            [{"k": "1"}, {"k": "2"}, {"k": "3"}],
            [[0.1], [0.2], [0.3]],
        )

    @patch("lsm.vectordb.migrations.chromadb_to_postgres.PostgreSQLProvider")
    @patch("lsm.vectordb.migrations.chromadb_to_postgres.ChromaDBProvider")
    def test_handles_empty_collection(self, mock_chroma_cls, mock_pg_cls) -> None:
        from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres
        from pathlib import Path

        mock_source = MagicMock()
        mock_chroma_cls.return_value = mock_source
        mock_source.count.return_value = 0
        mock_source.get.return_value = VectorDBGetResult(ids=[])

        mock_target = MagicMock()
        mock_pg_cls.return_value = mock_target

        pg_config = VectorDBConfig(provider="postgresql")
        result = migrate_chromadb_to_postgres(Path("/tmp/chroma"), "kb", pg_config)

        assert result["migrated"] == 0
        mock_target.add_chunks.assert_not_called()

    @patch("lsm.vectordb.migrations.chromadb_to_postgres.PostgreSQLProvider")
    @patch("lsm.vectordb.migrations.chromadb_to_postgres.ChromaDBProvider")
    def test_progress_callback_invoked(self, mock_chroma_cls, mock_pg_cls) -> None:
        from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres
        from pathlib import Path

        mock_source = MagicMock()
        mock_chroma_cls.return_value = mock_source
        mock_source.count.return_value = 5
        mock_source.get.side_effect = [
            VectorDBGetResult(
                ids=["a", "b"],
                documents=["d1", "d2"],
                metadatas=[{}, {}],
                embeddings=[[0.1], [0.2]],
            ),
            VectorDBGetResult(
                ids=["c", "d", "e"],
                documents=["d3", "d4", "d5"],
                metadatas=[{}, {}, {}],
                embeddings=[[0.3], [0.4], [0.5]],
            ),
            VectorDBGetResult(ids=[]),
        ]

        mock_target = MagicMock()
        mock_pg_cls.return_value = mock_target

        progress_calls = []
        pg_config = VectorDBConfig(provider="postgresql")
        result = migrate_chromadb_to_postgres(
            Path("/tmp/chroma"), "kb", pg_config, batch_size=2,
            progress_callback=lambda m, t: progress_calls.append((m, t)),
        )

        assert result["migrated"] == 5
        assert progress_calls == [(2, 5), (5, 5)]

    @patch("lsm.vectordb.migrations.chromadb_to_postgres.PostgreSQLProvider")
    @patch("lsm.vectordb.migrations.chromadb_to_postgres.ChromaDBProvider")
    def test_handles_numpy_embeddings_in_source_results(
        self,
        mock_chroma_cls,
        mock_pg_cls,
    ) -> None:
        from lsm.vectordb.migrations.chromadb_to_postgres import migrate_chromadb_to_postgres
        from pathlib import Path

        np = pytest.importorskip("numpy")

        mock_source = MagicMock()
        mock_chroma_cls.return_value = mock_source
        mock_source.count.return_value = 2
        mock_source.get.side_effect = [
            VectorDBGetResult(
                ids=["a", "b"],
                documents=["doc1", "doc2"],
                metadatas=[{"k": "1"}, {"k": "2"}],
                embeddings=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            ),
            VectorDBGetResult(ids=[]),
        ]

        mock_target = MagicMock()
        mock_pg_cls.return_value = mock_target

        pg_config = VectorDBConfig(provider="postgresql")
        result = migrate_chromadb_to_postgres(Path("/tmp/chroma"), "kb", pg_config, batch_size=100)

        assert result["migrated"] == 2
        mock_target.add_chunks.assert_called_once_with(
            ["a", "b"],
            ["doc1", "doc2"],
            [{"k": "1"}, {"k": "2"}],
            [[0.1, 0.2], [0.3, 0.4]],
        )
