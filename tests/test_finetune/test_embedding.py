"""Tests for embedding fine-tuning and model registry (Phase 15.4)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from lsm.finetune.embedding import TrainingPair, extract_training_pairs, finetune_embedding_model
from lsm.finetune.registry import (
    EmbeddingModelEntry,
    delete_model,
    get_active_model,
    list_models,
    register_model,
    set_active_model,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def conn():
    """In-memory SQLite database with schema."""
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS lsm_chunks (
            cid TEXT PRIMARY KEY,
            source_path TEXT,
            heading TEXT,
            chunk_text TEXT,
            chunk_index INTEGER,
            node_type TEXT DEFAULT 'chunk',
            is_current INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS lsm_embedding_models (
            model_id TEXT PRIMARY KEY,
            base_model TEXT,
            path TEXT,
            dimension INTEGER,
            created_at TEXT,
            is_active INTEGER DEFAULT 0
        );
    """)
    return db


def _insert_chunk(conn, cid, heading, text, source_path="/test.md"):
    conn.execute(
        "INSERT INTO lsm_chunks (cid, source_path, heading, chunk_text, chunk_index, node_type, is_current) "
        "VALUES (?, ?, ?, ?, 0, 'chunk', 1)",
        (cid, source_path, heading, text),
    )
    conn.commit()


# ------------------------------------------------------------------
# Tests: TrainingPair extraction
# ------------------------------------------------------------------


class TestExtractTrainingPairs:
    def test_extracts_pairs(self, conn):
        _insert_chunk(conn, "c1", "Introduction", "This is the introduction section with enough text to pass the minimum length filter.", "/doc.md")
        _insert_chunk(conn, "c2", "Methods", "This section describes the methods used in our analysis and their justification.", "/doc.md")

        pairs = extract_training_pairs(conn)
        assert len(pairs) == 2
        assert pairs[0].anchor == "Introduction"
        assert pairs[0].source_path == "/doc.md"
        assert len(pairs[0].positive) > 0

    def test_filters_short_content(self, conn):
        _insert_chunk(conn, "c1", "Title", "Short", "/doc.md")
        pairs = extract_training_pairs(conn, min_content_length=50)
        assert len(pairs) == 0

    def test_filters_null_heading(self, conn):
        _insert_chunk(conn, "c1", None, "A" * 100, "/doc.md")
        pairs = extract_training_pairs(conn)
        assert len(pairs) == 0

    def test_filters_empty_heading(self, conn):
        _insert_chunk(conn, "c1", "", "A" * 100, "/doc.md")
        pairs = extract_training_pairs(conn)
        assert len(pairs) == 0

    def test_max_pairs_limit(self, conn):
        for i in range(10):
            _insert_chunk(conn, f"c{i}", f"Heading {i}", "A" * 100, "/doc.md")

        pairs = extract_training_pairs(conn, max_pairs=3)
        assert len(pairs) == 3

    def test_truncates_positive_to_500(self, conn):
        _insert_chunk(conn, "c1", "Title", "A" * 1000, "/doc.md")
        pairs = extract_training_pairs(conn, min_content_length=50)
        assert len(pairs[0].positive) == 500

    def test_empty_corpus(self, conn):
        pairs = extract_training_pairs(conn)
        assert pairs == []

    def test_only_current_chunks(self, conn):
        conn.execute(
            "INSERT INTO lsm_chunks (cid, source_path, heading, chunk_text, chunk_index, node_type, is_current) "
            "VALUES (?, ?, ?, ?, 0, 'chunk', 0)",
            ("c1", "/doc.md", "Old Heading", "A" * 100),
        )
        conn.commit()
        pairs = extract_training_pairs(conn)
        assert len(pairs) == 0


# ------------------------------------------------------------------
# Tests: Model registry
# ------------------------------------------------------------------


class TestModelRegistry:
    def test_register_model(self, conn):
        entry = register_model(conn, "model-1", "base-model", "/path/to/model", 384)
        assert entry.model_id == "model-1"
        assert entry.base_model == "base-model"
        assert entry.path == "/path/to/model"
        assert entry.dimension == 384
        assert entry.is_active is False
        assert entry.created_at is not None

    def test_register_overwrites(self, conn):
        register_model(conn, "model-1", "base-v1", "/path/v1", 384)
        entry = register_model(conn, "model-1", "base-v2", "/path/v2", 768)
        assert entry.base_model == "base-v2"
        assert entry.dimension == 768

        models = list_models(conn)
        assert len(models) == 1

    def test_list_models_empty(self, conn):
        assert list_models(conn) == []

    def test_list_models_ordered_by_date(self, conn):
        register_model(conn, "model-a", "base", "/a", 384)
        register_model(conn, "model-b", "base", "/b", 384)
        models = list_models(conn)
        assert len(models) == 2
        # Most recent first
        assert models[0].model_id == "model-b"

    def test_set_active_model(self, conn):
        register_model(conn, "model-1", "base", "/path", 384)
        register_model(conn, "model-2", "base", "/path2", 384)

        set_active_model(conn, "model-1")
        active = get_active_model(conn)
        assert active is not None
        assert active.model_id == "model-1"
        assert active.is_active is True

    def test_set_active_deactivates_others(self, conn):
        register_model(conn, "model-1", "base", "/path", 384)
        register_model(conn, "model-2", "base", "/path2", 384)

        set_active_model(conn, "model-1")
        set_active_model(conn, "model-2")

        active = get_active_model(conn)
        assert active.model_id == "model-2"

        # model-1 should no longer be active
        models = list_models(conn)
        model_1 = [m for m in models if m.model_id == "model-1"][0]
        assert model_1.is_active is False

    def test_get_active_model_none(self, conn):
        assert get_active_model(conn) is None

    def test_delete_model(self, conn):
        register_model(conn, "model-1", "base", "/path", 384)
        assert delete_model(conn, "model-1") is True
        assert list_models(conn) == []

    def test_delete_nonexistent(self, conn):
        assert delete_model(conn, "nonexistent") is False


# ------------------------------------------------------------------
# Tests: finetune_embedding_model (mocked)
# ------------------------------------------------------------------


class TestFinetuneEmbeddingModel:
    def test_raises_on_empty_pairs(self):
        with pytest.raises(ValueError, match="No training pairs"):
            finetune_embedding_model([])

    def test_raises_on_missing_deps(self):
        pairs = [TrainingPair(anchor="Q", positive="A", source_path="/t.md")]
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError):
                finetune_embedding_model(pairs)

    @patch("lsm.finetune.embedding.SentenceTransformer", create=True)
    def test_finetune_success(self, mock_st_class):
        """Test successful fine-tuning with mocked sentence-transformers."""
        # Mock the SentenceTransformer model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        # Mock imports inside the function
        mock_input_example = MagicMock()
        mock_losses = MagicMock()
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=2)

        pairs = [
            TrainingPair(anchor="Heading 1", positive="Content 1", source_path="/a.md"),
            TrainingPair(anchor="Heading 2", positive="Content 2", source_path="/b.md"),
        ]

        with patch.dict("sys.modules", {
            "sentence_transformers": MagicMock(
                SentenceTransformer=mock_st_class,
                InputExample=mock_input_example,
                losses=mock_losses,
            ),
            "torch": MagicMock(),
            "torch.utils": MagicMock(),
            "torch.utils.data": MagicMock(DataLoader=MagicMock(return_value=mock_dataloader)),
        }):
            result = finetune_embedding_model(
                pairs,
                base_model="test-model",
                output_path="/tmp/test-output",
                epochs=1,
            )

        assert result["base_model"] == "test-model"
        assert result["output_path"] == "/tmp/test-output"
        assert result["dimension"] == 384
        assert result["num_pairs"] == 2
        assert result["epochs"] == 1
        assert "model_id" in result
        assert result["model_id"].startswith("finetuned-")
