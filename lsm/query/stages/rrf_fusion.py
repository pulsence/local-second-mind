"""
Reciprocal Rank Fusion (RRF) stage.

Combines dense and sparse retrieval results using the RRF formula:
    fused_score(d) = sum_over_lists( weight / (k + rank(d)) )

Reference: Cormack, Clarke, Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods."
"""

from __future__ import annotations

from typing import Dict, List, Optional

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate


def rrf_fuse(
    dense_results: List[Candidate],
    sparse_results: List[Candidate],
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    k: int = 60,
) -> List[Candidate]:
    """
    Fuse dense and sparse retrieval results using Reciprocal Rank Fusion.

    For each candidate appearing in either list, the fused score is:
        score = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)

    Candidates not appearing in a list get rank = len(list) + 1 as their
    default rank for that list.

    Args:
        dense_results: Candidates from dense recall, in ranked order.
        sparse_results: Candidates from sparse recall, in ranked order.
        dense_weight: Weight for dense retrieval scores.
        sparse_weight: Weight for sparse retrieval scores.
        k: RRF constant (default 60, per original paper).

    Returns:
        Merged candidates sorted by fused score (descending), each with
        ScoreBreakdown populated.
    """
    # Build rank maps
    dense_ranks: Dict[str, int] = {}
    dense_by_id: Dict[str, Candidate] = {}
    for rank, c in enumerate(dense_results, start=1):
        dense_ranks[c.cid] = rank
        dense_by_id[c.cid] = c

    sparse_ranks: Dict[str, int] = {}
    sparse_by_id: Dict[str, Candidate] = {}
    for rank, c in enumerate(sparse_results, start=1):
        sparse_ranks[c.cid] = rank
        sparse_by_id[c.cid] = c

    # Default rank for missing entries
    default_dense_rank = len(dense_results) + 1
    default_sparse_rank = len(sparse_results) + 1

    # Collect all unique candidate IDs
    all_ids = list(dict.fromkeys(
        [c.cid for c in dense_results] + [c.cid for c in sparse_results]
    ))

    # Compute fused scores
    fused: List[tuple] = []  # (fused_score, cid)
    for cid in all_ids:
        d_rank = dense_ranks.get(cid, default_dense_rank)
        s_rank = sparse_ranks.get(cid, default_sparse_rank)

        fused_score = (
            dense_weight / (k + d_rank)
            + sparse_weight / (k + s_rank)
        )
        fused.append((fused_score, cid))

    # Sort by fused score descending
    fused.sort(key=lambda x: x[0], reverse=True)

    # Build result candidates
    result: List[Candidate] = []
    for fused_score, cid in fused:
        # Prefer dense candidate (has embedding distance), fall back to sparse
        base = dense_by_id.get(cid) or sparse_by_id[cid]

        d_rank = dense_ranks.get(cid, default_dense_rank)
        s_rank = sparse_ranks.get(cid, default_sparse_rank)

        # Get scores from the original breakdowns
        dense_score = None
        sparse_score = None
        if cid in dense_by_id and dense_by_id[cid].score_breakdown:
            dense_score = dense_by_id[cid].score_breakdown.dense_score
        if cid in sparse_by_id and sparse_by_id[cid].score_breakdown:
            sparse_score = sparse_by_id[cid].score_breakdown.sparse_score

        breakdown = ScoreBreakdown(
            dense_score=dense_score,
            dense_rank=d_rank if cid in dense_ranks else None,
            sparse_score=sparse_score,
            sparse_rank=s_rank if cid in sparse_ranks else None,
            fused_score=fused_score,
        )

        result.append(
            Candidate(
                cid=base.cid,
                text=base.text,
                meta=base.meta,
                distance=base.distance,
                score_breakdown=breakdown,
            )
        )

    return result
