"""
Sparse (BM25/FTS5) recall stage.

Retrieves candidates via full-text search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

if TYPE_CHECKING:
    from lsm.vectordb.base import BaseVectorDBProvider


def sparse_recall(
    query_text: str,
    db: "BaseVectorDBProvider",
    top_k: int,
) -> List[Candidate]:
    """
    Retrieve candidates via BM25 full-text search.

    Each candidate gets a ScoreBreakdown with sparse_score and sparse_rank populated.

    Args:
        query_text: Query text for full-text matching.
        db: Vector database provider (must support fts_query).
        top_k: Number of candidates to retrieve.

    Returns:
        List of Candidates sorted by BM25 relevance.
    """
    result = db.fts_query(query_text, top_k)

    candidates = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(result.ids, result.documents, result.metadatas, result.distances),
        start=1,
    ):
        # dist is negated BM25 score (lower = better match)
        sparse_score = 1.0 / (rank + 1)  # Normalize to (0, 1) range
        candidates.append(
            Candidate(
                cid=cid,
                text=doc,
                meta=meta,
                distance=dist,
                score_breakdown=ScoreBreakdown(
                    sparse_score=sparse_score,
                    sparse_rank=rank,
                ),
            )
        )

    return candidates
