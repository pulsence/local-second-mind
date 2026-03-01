"""
Cluster-aware retrieval infrastructure.

Assigns ``cluster_id`` to all chunks via k-means or HDBSCAN, stores
centroids in ``lsm_cluster_centroids``, and supports cluster-filtered
retrieval via ``get_top_clusters()``.
"""

from __future__ import annotations

import logging
import sqlite3
from struct import pack, unpack
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core routines
# ------------------------------------------------------------------

def build_clusters(
    conn: sqlite3.Connection,
    algorithm: str = "kmeans",
    k: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Build cluster assignments for all current embeddings.

    Reads embeddings from ``vec_chunks``, runs the requested algorithm,
    writes ``cluster_id`` / ``cluster_size`` back to ``lsm_chunks``,
    and stores centroids in ``lsm_cluster_centroids``.

    Args:
        conn: Open SQLite connection (with sqlite-vec loaded).
        algorithm: ``"kmeans"`` or ``"hdbscan"``.
        k: Number of clusters (k-means only).
        random_state: Reproducibility seed.

    Returns:
        Summary dict with ``n_clusters``, ``n_chunks``, ``algorithm``.
    """
    # 1. Read all current embeddings
    rows = conn.execute(
        "SELECT chunk_id, embedding FROM vec_chunks WHERE is_current = 1"
    ).fetchall()

    if not rows:
        logger.warning("No current embeddings found — skipping clustering.")
        return {"n_clusters": 0, "n_chunks": 0, "algorithm": algorithm}

    chunk_ids: List[str] = []
    embeddings: List[List[float]] = []

    for row in rows:
        chunk_ids.append(str(row[0]))
        blob = bytes(row[1])
        dim = len(blob) // 4
        vec = list(unpack(f"{dim}f", blob))
        embeddings.append(vec)

    matrix = np.array(embeddings, dtype=np.float32)
    n_chunks = len(chunk_ids)

    # 2. Cluster
    if algorithm == "hdbscan":
        labels, centroids = _cluster_hdbscan(matrix)
    else:
        actual_k = min(k, n_chunks)
        labels, centroids = _cluster_kmeans(matrix, actual_k, random_state)

    n_clusters = len(centroids)

    # 3. Compute cluster sizes
    cluster_sizes: Dict[int, int] = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    # 4. Write centroids
    conn.execute("DELETE FROM lsm_cluster_centroids")
    for cluster_id, centroid in enumerate(centroids):
        blob = pack(f"{len(centroid)}f", *centroid)
        conn.execute(
            "INSERT INTO lsm_cluster_centroids (cluster_id, centroid, size) VALUES (?, ?, ?)",
            (cluster_id, blob, cluster_sizes.get(cluster_id, 0)),
        )

    # 5. Write cluster_id and cluster_size to lsm_chunks
    for chunk_id, label in zip(chunk_ids, labels):
        conn.execute(
            "UPDATE lsm_chunks SET cluster_id = ?, cluster_size = ? WHERE chunk_id = ?",
            (int(label), cluster_sizes.get(label, 0), chunk_id),
        )

    # 6. Update vec_chunks cluster_id
    for chunk_id, label in zip(chunk_ids, labels):
        conn.execute(
            "UPDATE vec_chunks SET cluster_id = ? WHERE chunk_id = ?",
            (int(label), chunk_id),
        )

    conn.commit()

    logger.info(
        "Clustering complete: %d clusters, %d chunks, algorithm=%s",
        n_clusters, n_chunks, algorithm,
    )

    return {
        "n_clusters": n_clusters,
        "n_chunks": n_chunks,
        "algorithm": algorithm,
    }


def get_top_clusters(
    query_embedding: List[float],
    conn: sqlite3.Connection,
    top_n: int = 5,
) -> List[int]:
    """Find the *top_n* clusters closest to the query embedding.

    Uses cosine similarity against stored centroids.

    Args:
        query_embedding: Query vector.
        conn: Open SQLite connection.
        top_n: Number of clusters to return.

    Returns:
        Sorted list of cluster IDs (most similar first).
    """
    rows = conn.execute(
        "SELECT cluster_id, centroid FROM lsm_cluster_centroids"
    ).fetchall()

    if not rows:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    scores: List[Tuple[int, float]] = []
    for row in rows:
        cluster_id = int(row[0])
        blob = bytes(row[1])
        dim = len(blob) // 4
        centroid = np.array(unpack(f"{dim}f", blob), dtype=np.float32)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            continue
        similarity = float(np.dot(query_vec, centroid) / (query_norm * centroid_norm))
        scores.append((cluster_id, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scores[:top_n]]


# ------------------------------------------------------------------
# Algorithm backends
# ------------------------------------------------------------------

def _cluster_kmeans(
    matrix: np.ndarray,
    k: int,
    random_state: int,
) -> Tuple[List[int], List[List[float]]]:
    """Run k-means clustering using sklearn."""
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(1024, len(matrix)),
        n_init="auto",
    )
    labels = kmeans.fit_predict(matrix)
    centroids = kmeans.cluster_centers_.tolist()
    return labels.tolist(), centroids


def _cluster_hdbscan(
    matrix: np.ndarray,
) -> Tuple[List[int], List[List[float]]]:
    """Run HDBSCAN clustering.

    Noise points (label -1) are reassigned to the nearest cluster.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        raise ImportError(
            "hdbscan is required for HDBSCAN clustering. "
            "Install it with: pip install hdbscan"
        )

    clusterer = HDBSCAN(min_cluster_size=5)
    raw_labels = clusterer.fit_predict(matrix)

    # Compute centroids for real clusters
    unique_labels = set(raw_labels)
    unique_labels.discard(-1)
    centroids: Dict[int, np.ndarray] = {}
    for label in sorted(unique_labels):
        mask = raw_labels == label
        centroids[label] = matrix[mask].mean(axis=0)

    # Reassign noise to nearest centroid
    if centroids and -1 in set(raw_labels):
        centroid_matrix = np.array(list(centroids.values()))
        centroid_labels = list(centroids.keys())
        noise_mask = raw_labels == -1
        noise_vecs = matrix[noise_mask]
        if len(noise_vecs) > 0:
            # Cosine similarity
            norms_n = np.linalg.norm(noise_vecs, axis=1, keepdims=True)
            norms_c = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
            norms_n = np.maximum(norms_n, 1e-10)
            norms_c = np.maximum(norms_c, 1e-10)
            similarities = (noise_vecs / norms_n) @ (centroid_matrix / norms_c).T
            nearest = similarities.argmax(axis=1)
            raw_labels[noise_mask] = np.array([centroid_labels[i] for i in nearest])

    # Recompute centroids including reassigned points
    final_labels = list(set(raw_labels))
    final_centroids: List[List[float]] = []
    label_map: Dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(final_labels)):
        label_map[old_id] = new_id
        mask = raw_labels == old_id
        final_centroids.append(matrix[mask].mean(axis=0).tolist())

    mapped_labels = [label_map[int(l)] for l in raw_labels]
    return mapped_labels, final_centroids
