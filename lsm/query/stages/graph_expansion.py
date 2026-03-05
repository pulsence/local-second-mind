"""
Graph-augmented retrieval stage.

After initial vector retrieval, expands the candidate set using knowledge
graph traversal. Expanded nodes receive a decaying graph_expansion_score
based on hop distance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set

from lsm.db.compat import fetchall
from lsm.db.connection import resolve_connection
from lsm.db.tables import DEFAULT_TABLE_NAMES
from lsm.query.pipeline_types import ScoreBreakdown
from lsm.query.session import Candidate

if TYPE_CHECKING:
    from lsm.vectordb.base import BaseVectorDBProvider


def expand_via_graph(
    candidates: List[Candidate],
    db: "BaseVectorDBProvider",
    max_hops: int = 2,
    edge_types: Optional[List[str]] = None,
    expansion_k: int = 5,
) -> List[Candidate]:
    """Expand candidates using knowledge graph traversal.

    For each candidate, finds related nodes via graph traversal and
    retrieves their chunks. Expanded candidates get a decaying
    graph_expansion_score.

    Args:
        candidates: Initial retrieval candidates.
        db: Vector database provider with graph support.
        max_hops: Maximum traversal depth.
        edge_types: Optional edge type filter (e.g., ["contains", "references"]).
        expansion_k: Max chunks to retrieve per expanded source.

    Returns:
        Original candidates + graph-expanded candidates (deduplicated).
    """
    if not candidates:
        return candidates

    # Collect source paths from existing candidates
    existing_cids: Set[str] = {c.cid for c in candidates}
    source_paths: Set[str] = set()
    for c in candidates:
        sp = (c.meta or {}).get("source_path", "")
        if sp:
            source_paths.add(sp)

    # Build start node IDs from candidate source paths
    # Use the graph_builder's stable_id convention
    import hashlib

    def _stable_id(source_path: str, *parts: str) -> str:
        raw = "|".join([source_path] + list(parts))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    start_ids = [_stable_id(sp, "file") for sp in source_paths]
    if not start_ids:
        return candidates

    # Graph traversal
    try:
        reachable_ids = db.graph_traverse(
            start_ids, max_hops=max_hops, edge_types=edge_types,
        )
    except Exception:
        return candidates

    if not reachable_ids:
        return candidates

    # Find new source paths from reachable graph nodes
    new_source_paths: Set[str] = set()
    try:
        with resolve_connection(db) as conn:
            placeholders = ", ".join(["?"] * len(reachable_ids))
            rows = fetchall(
                conn,
                f"SELECT DISTINCT source_path FROM {DEFAULT_TABLE_NAMES.graph_nodes} "
                f"WHERE node_id IN ({placeholders}) AND source_path != ''",
                reachable_ids,
            )
        for row in rows:
            sp = row[0]
            if sp and sp not in source_paths:
                new_source_paths.add(sp)
    except Exception:
        return candidates

    if not new_source_paths:
        return candidates

    # Retrieve chunks from expanded source paths
    expanded: List[Candidate] = []
    for sp in new_source_paths:
        try:
            result = db.get(
                filters={"source_path": sp, "is_current": True, "node_type": "chunk"},
                limit=expansion_k,
                include=["documents", "metadatas"],
            )
            for i, cid in enumerate(result.ids):
                if cid in existing_cids:
                    continue
                existing_cids.add(cid)

                doc = result.documents[i] if result.documents else ""
                meta = result.metadatas[i] if result.metadatas else {}

                expanded.append(
                    Candidate(
                        cid=cid,
                        text=doc,
                        meta=meta,
                        distance=None,
                        score_breakdown=ScoreBreakdown(
                            graph_expansion_score=1.0 / (1 + len(expanded)),
                        ),
                    )
                )
        except Exception:
            continue

    return candidates + expanded
