"""
Dense (vector) recall stage.

Retrieves candidates via embedding similarity search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

if TYPE_CHECKING:
    from lsm.vectordb.base import BaseVectorDBProvider


def dense_recall(
    query_embedding: List[float],
    db: "BaseVectorDBProvider",
    top_k: int,
    filters: Optional[dict] = None,
) -> List[Candidate]:
    """
    Retrieve candidates via dense vector similarity search.

    Each candidate gets a ScoreBreakdown with dense_score and dense_rank populated.

    Args:
        query_embedding: Query vector.
        db: Vector database provider.
        top_k: Number of candidates to retrieve.
        filters: Optional metadata filters.

    Returns:
        List of Candidates sorted by vector distance (ascending).
    """
    result = db.query(query_embedding, top_k, filters=filters)

    candidates = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(result.ids, result.documents, result.metadatas, result.distances),
        start=1,
    ):
        score = 1.0 - (dist or 0.0)
        candidates.append(
            Candidate(
                cid=cid,
                text=doc,
                meta=meta,
                distance=dist,
                score_breakdown=ScoreBreakdown(
                    dense_score=score,
                    dense_rank=rank,
                ),
            )
        )

    return candidates
