"""Tests for cross-encoder reranking stage."""

from __future__ import annotations

import pytest

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate
from lsm.query.stages.cross_encoder import CrossEncoderReranker


def _candidate(cid, text="", distance=0.2, dense_score=None, dense_rank=None):
    bd = ScoreBreakdown(dense_score=dense_score, dense_rank=dense_rank) if dense_score else None
    return Candidate(cid=cid, text=text, meta={}, distance=distance, score_breakdown=bd)


class FakeCrossEncoder:
    """Fake CrossEncoder that returns scores proportional to text length."""

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device

    def predict(self, pairs):
        # Score by text length — longer text = higher score
        return [len(pair[1]) / 100.0 for pair in pairs]


class TestCrossEncoderReranker:
    def test_empty_candidates(self):
        reranker = CrossEncoderReranker()
        result = reranker.rerank("test", [], top_k=5)
        assert result == []

    def test_reranking_changes_order(self, monkeypatch):
        """Cross-encoder should reorder candidates by its score."""
        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: FakeCrossEncoder(),
        )
        reranker = CrossEncoderReranker()
        candidates = [
            _candidate("short", text="hi"),
            _candidate("long", text="this is a much longer text for scoring"),
            _candidate("medium", text="medium length"),
        ]
        result = reranker.rerank("query", candidates, top_k=3)
        # Longest text should be first (highest score)
        assert result[0].cid == "long"
        assert result[1].cid == "medium"
        assert result[2].cid == "short"

    def test_top_k_limits_results(self, monkeypatch):
        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: FakeCrossEncoder(),
        )
        reranker = CrossEncoderReranker()
        candidates = [_candidate(f"c{i}", text=f"text {i}") for i in range(10)]
        result = reranker.rerank("query", candidates, top_k=3)
        assert len(result) == 3

    def test_rerank_score_populated(self, monkeypatch):
        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: FakeCrossEncoder(),
        )
        reranker = CrossEncoderReranker()
        candidates = [_candidate("a", text="some text")]
        result = reranker.rerank("query", candidates, top_k=1)
        assert result[0].score_breakdown is not None
        assert result[0].score_breakdown.rerank_score is not None
        assert result[0].score_breakdown.rerank_score > 0

    def test_preserves_dense_score(self, monkeypatch):
        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: FakeCrossEncoder(),
        )
        reranker = CrossEncoderReranker()
        candidates = [_candidate("a", text="text", dense_score=0.9, dense_rank=1)]
        result = reranker.rerank("query", candidates, top_k=1)
        assert result[0].score_breakdown.dense_score == 0.9
        assert result[0].score_breakdown.dense_rank == 1

    def test_graceful_degradation_no_model(self, monkeypatch):
        """When model can't load, returns dense recall order."""
        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: None,
        )
        reranker = CrossEncoderReranker()
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].cid == "a"
        assert result[1].cid == "b"

    def test_graceful_degradation_predict_error(self, monkeypatch):
        """When predict() raises, returns dense recall order."""

        class FailingEncoder:
            def predict(self, pairs):
                raise RuntimeError("CUDA OOM")

        monkeypatch.setattr(
            "lsm.query.stages.cross_encoder.CrossEncoderReranker._load_model",
            lambda self: FailingEncoder(),
        )
        reranker = CrossEncoderReranker()
        candidates = [_candidate("a"), _candidate("b")]
        result = reranker.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].cid == "a"

    def test_device_parameter_stored(self):
        reranker = CrossEncoderReranker(device="cuda:0")
        assert reranker.device == "cuda:0"

    def test_model_name_stored(self):
        reranker = CrossEncoderReranker(model_name="custom/model")
        assert reranker.model_name == "custom/model"
