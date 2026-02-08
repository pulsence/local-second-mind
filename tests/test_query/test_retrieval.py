"""
Tests for query retrieval module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lsm.query.retrieval import (
    init_embedder,
    embed_text,
    retrieve_candidates,
    filter_candidates,
    compute_relevance,
    _normalize_ext,
)
from lsm.query.session import Candidate
from lsm.vectordb.base import VectorDBQueryResult


class TestEmbedderInit:
    """Tests for embedder initialization."""

    @patch("lsm.query.retrieval.SentenceTransformer")
    def test_init_embedder_cpu(self, mock_st_class, monkeypatch):
        """Test initializing embedder on CPU."""
        import lsm.query.retrieval as retrieval
        monkeypatch.setattr(retrieval, "_sentence_transformer_import_error", None)
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        embedder = init_embedder("sentence-transformers/all-MiniLM-L6-v2", "cpu")

        mock_st_class.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        assert embedder == mock_model

    @patch("lsm.query.retrieval.SentenceTransformer")
    def test_init_embedder_cuda(self, mock_st_class, monkeypatch):
        """Test initializing embedder on CUDA."""
        import lsm.query.retrieval as retrieval
        monkeypatch.setattr(retrieval, "_sentence_transformer_import_error", None)
        mock_model = Mock()
        mock_st_class.return_value = mock_model

        embedder = init_embedder("test-model", "cuda")

        mock_st_class.assert_called_once_with("test-model", device="cuda")

    def test_init_embedder_import_error(self, monkeypatch):
        """Test clear error when sentence-transformers import failed."""
        import lsm.query.retrieval as retrieval

        monkeypatch.setattr(retrieval, "_sentence_transformer_import_error", ImportError("missing dep"))
        with pytest.raises(RuntimeError, match="Failed to import sentence-transformers"):
            retrieval.init_embedder("test-model", "cpu")

    @patch("lsm.query.retrieval.SentenceTransformer")
    def test_init_embedder_cuda_falls_back_to_cpu(self, mock_st_class, monkeypatch):
        """If CUDA isn't available, init should retry on CPU."""
        import lsm.query.retrieval as retrieval

        monkeypatch.setattr(retrieval, "_sentence_transformer_import_error", None)
        mock_model = Mock()
        mock_st_class.side_effect = [RuntimeError("Torch not compiled with CUDA enabled"), mock_model]

        embedder = init_embedder("test-model", "cuda")

        assert embedder == mock_model
        assert mock_st_class.call_count == 2
        assert mock_st_class.call_args_list[0].kwargs["device"] == "cuda"
        assert mock_st_class.call_args_list[1].kwargs["device"] == "cpu"


class TestEmbedText:
    """Tests for text embedding."""

    def test_embed_text(self):
        """Test embedding text to vector."""
        mock_model = Mock()
        mock_vector = [[0.1, 0.2, 0.3, 0.4]]
        mock_model.encode.return_value = mock_vector

        result = embed_text(mock_model, "What is Python?", batch_size=32)

        mock_model.encode.assert_called_once_with(
            ["What is Python?"],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        assert result == [0.1, 0.2, 0.3, 0.4]

    def test_embed_text_custom_batch_size(self):
        """Test embedding with custom batch size."""
        mock_model = Mock()
        mock_vector = [[0.1, 0.2]]
        mock_model.encode.return_value = mock_vector

        result = embed_text(mock_model, "Test", batch_size=64)

        assert mock_model.encode.call_args[1]["batch_size"] == 64


class TestRetrieveCandidates:
    """Tests for candidate retrieval via provider interface."""

    def test_retrieve_candidates(self):
        """Test retrieving candidates from vector DB provider."""
        mock_provider = Mock()
        mock_provider.query.return_value = VectorDBQueryResult(
            ids=["id1", "id2", "id3"],
            documents=["Text 1", "Text 2", "Text 3"],
            metadatas=[
                {"source_path": "/docs/a.md", "chunk_index": 0},
                {"source_path": "/docs/b.md", "chunk_index": 0},
                {"source_path": "/docs/a.md", "chunk_index": 1},
            ],
            distances=[0.1, 0.2, 0.3],
        )

        query_vector = [0.1, 0.2, 0.3]
        candidates = retrieve_candidates(mock_provider, query_vector, k=3)

        mock_provider.query.assert_called_once_with(
            query_vector, top_k=3, filters=None,
        )

        assert len(candidates) == 3
        assert candidates[0].cid == "id1"
        assert candidates[0].text == "Text 1"
        assert candidates[0].distance == 0.1
        assert candidates[0].meta["source_path"] == "/docs/a.md"

    def test_retrieve_candidates_empty_results(self):
        """Test retrieving when no results found."""
        mock_provider = Mock()
        mock_provider.query.return_value = VectorDBQueryResult(
            ids=[], documents=[], metadatas=[], distances=[],
        )

        query_vector = [0.1, 0.2]
        candidates = retrieve_candidates(mock_provider, query_vector, k=5)

        assert candidates == []

    def test_retrieve_candidates_with_where_filter(self):
        """Test that where_filter is passed through."""
        mock_provider = Mock()
        mock_provider.query.return_value = VectorDBQueryResult(
            ids=["id1"], documents=["Text"], metadatas=[{}], distances=[0.1],
        )

        retrieve_candidates(
            mock_provider, [0.1], k=5, where_filter={"is_current": True},
        )

        mock_provider.query.assert_called_once_with(
            [0.1], top_k=5, filters={"is_current": True},
        )


class TestNormalizeExt:
    """Tests for extension normalization."""

    def test_normalize_ext_with_dot(self):
        """Test normalizing extension that already has dot."""
        assert _normalize_ext(".md") == ".md"
        assert _normalize_ext(".PDF") == ".pdf"

    def test_normalize_ext_without_dot(self):
        """Test normalizing extension without dot."""
        assert _normalize_ext("md") == ".md"
        assert _normalize_ext("TXT") == ".txt"

    def test_normalize_ext_empty(self):
        """Test normalizing empty extension."""
        assert _normalize_ext("") == ""
        assert _normalize_ext("  ") == ""

    def test_normalize_ext_none(self):
        """Test normalizing None extension."""
        assert _normalize_ext(None) == ""  # type: ignore


class TestFilterCandidates:
    """Tests for candidate filtering."""

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for testing."""
        return [
            Candidate(
                cid="1",
                text="Python docs",
                meta={"source_path": "/docs/python.md", "ext": ".md"},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Java docs",
                meta={"source_path": "/docs/java.pdf", "ext": ".pdf"},
                distance=0.2,
            ),
            Candidate(
                cid="3",
                text="Python guide",
                meta={"source_path": "/guides/python.txt", "ext": ".txt"},
                distance=0.15,
            ),
            Candidate(
                cid="4",
                text="Notes",
                meta={"source_path": "/notes/misc.md", "ext": ".md"},
                distance=0.25,
            ),
        ]

    def test_filter_no_filters(self, sample_candidates):
        """Test filtering with no filters returns all candidates."""
        result = filter_candidates(sample_candidates)
        assert len(result) == 4

    def test_filter_by_path_contains_string(self, sample_candidates):
        """Test filtering by path contains (single string)."""
        result = filter_candidates(
            sample_candidates,
            path_contains="python"
        )
        assert len(result) == 2
        assert all("python" in c.source_path.lower() for c in result)

    def test_filter_by_path_contains_list(self, sample_candidates):
        """Test filtering by path contains (list of strings)."""
        result = filter_candidates(
            sample_candidates,
            path_contains=["docs", "guides"]
        )
        assert len(result) == 3
        paths = [c.source_path for c in result]
        assert "/notes/misc.md" not in paths

    def test_filter_by_ext_allow(self, sample_candidates):
        """Test filtering by allowed extensions."""
        result = filter_candidates(
            sample_candidates,
            ext_allow=[".md", ".txt"]
        )
        assert len(result) == 3
        for c in result:
            assert c.ext in [".md", ".txt"]

    def test_filter_by_ext_deny(self, sample_candidates):
        """Test filtering by denied extensions."""
        result = filter_candidates(
            sample_candidates,
            ext_deny=[".pdf"]
        )
        assert len(result) == 3
        assert all(c.ext != ".pdf" for c in result)

    def test_filter_combined(self, sample_candidates):
        """Test combining multiple filters."""
        result = filter_candidates(
            sample_candidates,
            path_contains="python",
            ext_allow=[".md", ".txt"]
        )
        assert len(result) == 2
        assert all("python" in c.source_path.lower() for c in result)
        assert all(c.ext in [".md", ".txt"] for c in result)

    def test_filter_empty_candidates(self):
        """Test filtering empty candidate list."""
        result = filter_candidates([], path_contains="test")
        assert result == []

    def test_filter_no_matches(self, sample_candidates):
        """Test filtering when no candidates match."""
        result = filter_candidates(
            sample_candidates,
            path_contains="nonexistent"
        )
        assert result == []

    def test_filter_ext_normalization(self, sample_candidates):
        """Test that extensions are normalized during filtering."""
        result = filter_candidates(
            sample_candidates,
            ext_allow=["md", "PDF"]  # Without dots, mixed case
        )
        assert len(result) == 3
        exts = {c.ext for c in result}
        assert ".md" in exts
        assert ".pdf" in exts


class TestComputeRelevance:
    """Tests for relevance computation."""

    def test_compute_relevance_with_candidates(self):
        """Test computing relevance from candidates."""
        candidates = [
            Candidate(cid="1", text="A", meta={}, distance=0.1),
            Candidate(cid="2", text="B", meta={}, distance=0.2),
            Candidate(cid="3", text="C", meta={}, distance=0.15),
        ]

        relevance = compute_relevance(candidates)

        # Best distance is 0.1, so relevance = 1.0 - 0.1 = 0.9
        assert relevance == 0.9

    def test_compute_relevance_empty_list(self):
        """Test computing relevance with empty list."""
        relevance = compute_relevance([])
        assert relevance == -1.0

    def test_compute_relevance_no_distances(self):
        """Test computing relevance when no distances present."""
        candidates = [
            Candidate(cid="1", text="A", meta={}, distance=None),
            Candidate(cid="2", text="B", meta={}, distance=None),
        ]

        relevance = compute_relevance(candidates)
        assert relevance == -1.0

    def test_compute_relevance_clamping_high(self):
        """Test relevance clamping for very low distances."""
        candidates = [
            Candidate(cid="1", text="A", meta={}, distance=-0.5),  # Would give 1.5
        ]

        relevance = compute_relevance(candidates)
        assert relevance == 1.0  # Clamped to 1.0

    def test_compute_relevance_clamping_low(self):
        """Test relevance clamping for very high distances."""
        candidates = [
            Candidate(cid="1", text="A", meta={}, distance=2.5),  # Would give -1.5
        ]

        relevance = compute_relevance(candidates)
        assert relevance == -1.0  # Clamped to -1.0

    def test_compute_relevance_mixed_distances(self):
        """Test computing relevance with some None distances."""
        candidates = [
            Candidate(cid="1", text="A", meta={}, distance=None),
            Candidate(cid="2", text="B", meta={}, distance=0.3),
            Candidate(cid="3", text="C", meta={}, distance=0.2),
        ]

        relevance = compute_relevance(candidates)
        # Best non-None distance is 0.2
        assert relevance == 0.8
