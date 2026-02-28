"""Tests for RRF fusion stage."""

from __future__ import annotations

import pytest

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate
from lsm.query.stages.rrf_fusion import rrf_fuse


def _candidate(cid: str, text: str = "", distance: float = 0.0, breakdown=None):
    return Candidate(cid=cid, text=text, meta={}, distance=distance, score_breakdown=breakdown)


# ------------------------------------------------------------------
# Basic fusion
# ------------------------------------------------------------------


class TestRRFFuseBasic:
    def test_empty_inputs(self):
        result = rrf_fuse([], [])
        assert result == []

    def test_dense_only_input(self):
        dense = [_candidate("a"), _candidate("b")]
        result = rrf_fuse(dense, [])
        assert len(result) == 2
        assert result[0].cid == "a"
        assert result[1].cid == "b"

    def test_sparse_only_input(self):
        sparse = [_candidate("x"), _candidate("y")]
        result = rrf_fuse([], sparse)
        assert len(result) == 2
        assert result[0].cid == "x"
        assert result[1].cid == "y"

    def test_identical_lists(self):
        dense = [_candidate("a"), _candidate("b")]
        sparse = [_candidate("a"), _candidate("b")]
        result = rrf_fuse(dense, sparse)
        assert len(result) == 2
        # Both are in both lists — "a" is rank 1 in both, should rank first
        assert result[0].cid == "a"
        assert result[1].cid == "b"


# ------------------------------------------------------------------
# Ranking correctness
# ------------------------------------------------------------------


class TestRRFRanking:
    def test_overlapping_candidates_ranked_higher(self):
        """A candidate appearing in both lists should score higher than one in only one."""
        dense = [_candidate("shared"), _candidate("dense_only")]
        sparse = [_candidate("shared"), _candidate("sparse_only")]
        result = rrf_fuse(dense, sparse)
        assert result[0].cid == "shared"

    def test_rank_ordering_with_disjoint_lists(self):
        """With fully disjoint lists, dense rank-1 and sparse rank-1 should be top two."""
        dense = [_candidate("d1"), _candidate("d2"), _candidate("d3")]
        sparse = [_candidate("s1"), _candidate("s2"), _candidate("s3")]
        result = rrf_fuse(dense, sparse, dense_weight=0.5, sparse_weight=0.5)
        # d1 and s1 should both be rank-1 in their respective lists
        # With equal weights, the top two should be d1 and s1
        top_ids = {result[0].cid, result[1].cid}
        assert top_ids == {"d1", "s1"}

    def test_dense_weight_dominance(self):
        """With high dense weight, dense-only candidates rank above sparse-only."""
        dense = [_candidate("d1")]
        sparse = [_candidate("s1")]
        result = rrf_fuse(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
        assert result[0].cid == "d1"

    def test_sparse_weight_dominance(self):
        """With high sparse weight, sparse-only candidates rank above dense-only."""
        dense = [_candidate("d1")]
        sparse = [_candidate("s1")]
        result = rrf_fuse(dense, sparse, dense_weight=0.1, sparse_weight=0.9)
        assert result[0].cid == "s1"


# ------------------------------------------------------------------
# ScoreBreakdown
# ------------------------------------------------------------------


class TestRRFScoreBreakdown:
    def test_fused_score_populated(self):
        dense = [_candidate("a")]
        sparse = [_candidate("b")]
        result = rrf_fuse(dense, sparse)
        for c in result:
            assert c.score_breakdown is not None
            assert c.score_breakdown.fused_score is not None
            assert c.score_breakdown.fused_score > 0

    def test_dense_rank_populated_for_dense_candidate(self):
        dense = [_candidate("a")]
        result = rrf_fuse(dense, [])
        assert result[0].score_breakdown.dense_rank == 1

    def test_sparse_rank_populated_for_sparse_candidate(self):
        sparse = [_candidate("a")]
        result = rrf_fuse([], sparse)
        assert result[0].score_breakdown.sparse_rank == 1

    def test_both_ranks_for_overlapping_candidate(self):
        dense = [_candidate("a"), _candidate("b")]
        sparse = [_candidate("b"), _candidate("a")]
        result = rrf_fuse(dense, sparse)
        by_id = {c.cid: c for c in result}
        # "a" is dense_rank=1, sparse_rank=2
        assert by_id["a"].score_breakdown.dense_rank == 1
        assert by_id["a"].score_breakdown.sparse_rank == 2
        # "b" is dense_rank=2, sparse_rank=1
        assert by_id["b"].score_breakdown.dense_rank == 2
        assert by_id["b"].score_breakdown.sparse_rank == 1

    def test_missing_list_rank_is_none(self):
        dense = [_candidate("d")]
        sparse = [_candidate("s")]
        result = rrf_fuse(dense, sparse)
        by_id = {c.cid: c for c in result}
        assert by_id["d"].score_breakdown.dense_rank == 1
        assert by_id["d"].score_breakdown.sparse_rank is None
        assert by_id["s"].score_breakdown.sparse_rank == 1
        assert by_id["s"].score_breakdown.dense_rank is None

    def test_preserves_original_dense_score(self):
        bd = ScoreBreakdown(dense_score=0.95, dense_rank=1)
        dense = [_candidate("a", breakdown=bd)]
        result = rrf_fuse(dense, [])
        assert result[0].score_breakdown.dense_score == 0.95

    def test_preserves_original_sparse_score(self):
        bd = ScoreBreakdown(sparse_score=0.8, sparse_rank=1)
        sparse = [_candidate("a", breakdown=bd)]
        result = rrf_fuse([], sparse)
        assert result[0].score_breakdown.sparse_score == 0.8


# ------------------------------------------------------------------
# RRF formula correctness
# ------------------------------------------------------------------


class TestRRFFormula:
    def test_formula_single_candidate_in_both(self):
        """Verify exact RRF formula: score = dw/(k+dr) + sw/(k+sr)."""
        dense = [_candidate("a")]
        sparse = [_candidate("a")]
        k_val = 60
        dw, sw = 0.7, 0.3
        result = rrf_fuse(dense, sparse, dense_weight=dw, sparse_weight=sw, k=k_val)
        expected = dw / (k_val + 1) + sw / (k_val + 1)
        assert abs(result[0].score_breakdown.fused_score - expected) < 1e-10

    def test_formula_different_ranks(self):
        """Verify formula with different dense and sparse ranks."""
        dense = [_candidate("a"), _candidate("b")]
        sparse = [_candidate("b"), _candidate("a")]
        k_val = 60
        dw, sw = 0.7, 0.3
        result = rrf_fuse(dense, sparse, dense_weight=dw, sparse_weight=sw, k=k_val)
        by_id = {c.cid: c for c in result}
        # "a": dense_rank=1, sparse_rank=2
        expected_a = dw / (k_val + 1) + sw / (k_val + 2)
        assert abs(by_id["a"].score_breakdown.fused_score - expected_a) < 1e-10
        # "b": dense_rank=2, sparse_rank=1
        expected_b = dw / (k_val + 2) + sw / (k_val + 1)
        assert abs(by_id["b"].score_breakdown.fused_score - expected_b) < 1e-10

    def test_formula_missing_from_one_list(self):
        """Candidate in only one list uses default rank = len(list) + 1."""
        dense = [_candidate("d1"), _candidate("d2")]
        sparse = [_candidate("s1")]
        k_val = 60
        dw, sw = 0.7, 0.3
        result = rrf_fuse(dense, sparse, dense_weight=dw, sparse_weight=sw, k=k_val)
        by_id = {c.cid: c for c in result}
        # "d1": dense_rank=1, not in sparse → default_sparse_rank = 2 (len(sparse)+1)
        expected_d1 = dw / (k_val + 1) + sw / (k_val + 2)
        assert abs(by_id["d1"].score_breakdown.fused_score - expected_d1) < 1e-10
        # "s1": sparse_rank=1, not in dense → default_dense_rank = 3 (len(dense)+1)
        expected_s1 = dw / (k_val + 3) + sw / (k_val + 1)
        assert abs(by_id["s1"].score_breakdown.fused_score - expected_s1) < 1e-10

    def test_custom_k_value(self):
        """Different k constant affects scores."""
        dense = [_candidate("a")]
        sparse = [_candidate("a")]
        result_k10 = rrf_fuse(dense, sparse, k=10)
        result_k100 = rrf_fuse(dense, sparse, k=100)
        # Smaller k → higher individual scores
        assert result_k10[0].score_breakdown.fused_score > result_k100[0].score_breakdown.fused_score


# ------------------------------------------------------------------
# Candidate data integrity
# ------------------------------------------------------------------


class TestRRFCandidateData:
    def test_text_preserved_from_dense(self):
        dense = [_candidate("a", text="dense text", distance=0.1)]
        sparse = [_candidate("a", text="sparse text", distance=0.2)]
        result = rrf_fuse(dense, sparse)
        # Dense candidate is preferred when present in both
        assert result[0].text == "dense text"
        assert result[0].distance == 0.1

    def test_sparse_fallback_when_not_in_dense(self):
        sparse = [_candidate("s1", text="from sparse")]
        result = rrf_fuse([], sparse)
        assert result[0].text == "from sparse"

    def test_deduplication(self):
        """Same CID appearing in both lists produces single result."""
        dense = [_candidate("dup")]
        sparse = [_candidate("dup")]
        result = rrf_fuse(dense, sparse)
        assert len(result) == 1
        assert result[0].cid == "dup"

    def test_order_stability(self):
        """Result maintains stable ordering for equal fused scores."""
        dense = [_candidate("a"), _candidate("b"), _candidate("c")]
        sparse = [_candidate("a"), _candidate("b"), _candidate("c")]
        result = rrf_fuse(dense, sparse)
        # All have same rank in both lists, so order follows input order
        assert [c.cid for c in result] == ["a", "b", "c"]
