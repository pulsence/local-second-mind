"""
Cluster-aware retrieval infrastructure.

Assigns ``cluster_id`` to all chunks via k-means or HDBSCAN, stores
centroids in ``lsm_cluster_centroids``, and supports cluster-filtered
retrieval via ``get_top_clusters()``.
"""

from __future__ import annotations

import logging
from struct import pack, unpack
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lsm.db.compat import commit, execute, fetchall, fetchone, is_sqlite, table_exists
from lsm.db.connection import resolve_connection
from lsm.db.tables import TableNames, DEFAULT_TABLE_NAMES

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core routines
# ------------------------------------------------------------------

def build_clusters(
    db_or_conn: Any,
    algorithm: str = "kmeans",
    k: int = 50,
    random_state: int = 42,
    table_names: TableNames | None = None,
) -> Dict[str, Any]:
    """Build cluster assignments for all current embeddings.

    Uses the provider embedding API when available, otherwise falls back
    to reading SQLite ``vec_chunks`` directly, runs the requested algorithm,
    writes ``cluster_id`` / ``cluster_size`` back to ``lsm_chunks``,
    and stores centroids in ``lsm_cluster_centroids``.

    Args:
        db_or_conn: Vector provider (preferred) or raw DB connection.
        algorithm: ``"kmeans"`` or ``"hdbscan"``.
        k: Number of clusters (k-means only).
        random_state: Reproducibility seed.

    Returns:
        Summary dict with ``n_clusters``, ``n_chunks``, ``algorithm``.
    """
    tn = table_names or DEFAULT_TABLE_NAMES
    provider = db_or_conn if hasattr(db_or_conn, "get_embeddings") else None

    chunk_ids: List[str] = []
    embeddings: List[List[float]] = []

    if provider is not None:
        logger.info("Cluster build: reading embeddings from provider API...")
        try:
            chunk_ids, embeddings = provider.get_embeddings(only_current=True)
        except NotImplementedError:
            provider = None

    if provider is None:
        conn = db_or_conn
        if not is_sqlite(conn):
            raise ValueError(
                "build_clusters requires a vector provider for non-SQLite backends."
            )
        if not table_exists(conn, tn.vec_chunks):
            logger.warning("Cluster build skipped: %s does not exist.", tn.vec_chunks)
            return {"n_clusters": 0, "n_chunks": 0, "algorithm": algorithm}
        logger.info("Cluster build: reading embeddings from %s...", tn.vec_chunks)
        rows = fetchall(
            conn,
            f"SELECT chunk_id, embedding FROM {tn.vec_chunks} WHERE is_current = 1",
        )
        if rows:
            _sqlite = is_sqlite(conn)
            for i, row in enumerate(rows, 1):
                chunk_ids.append(str(row[0]))
                raw = row[1]
                if _sqlite:
                    blob = bytes(raw)
                    dim = len(blob) // 4
                    vec = list(unpack(f"{dim}f", blob))
                else:
                    vec = list(raw) if not isinstance(raw, list) else raw
                embeddings.append(vec)
                if i % 10000 == 0 or i == len(rows):
                    logger.info("Cluster build: unpacked %s/%s embeddings.", f"{i:,}", f"{len(rows):,}")

    if not chunk_ids or not embeddings:
        logger.warning("No current embeddings found — skipping clustering.")
        return {"n_clusters": 0, "n_chunks": 0, "algorithm": algorithm}

    matrix = np.array(embeddings, dtype=np.float32)
    n_chunks = len(chunk_ids)

    # 2. Cluster
    logger.info("Cluster build: running %s (k=%d) on %s vectors...", algorithm, k, f"{n_chunks:,}")
    if algorithm == "hdbscan":
        labels, centroids = _cluster_hdbscan(matrix)
    else:
        actual_k = min(k, n_chunks)
        labels, centroids = _cluster_kmeans(matrix, actual_k, random_state)

    n_clusters = len(centroids)
    logger.info("Cluster build: %d clusters computed.", n_clusters)

    # 3. Compute cluster sizes
    cluster_sizes: Dict[int, int] = {}
    for label in labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    updates = [(chunk_id, int(label)) for chunk_id, label in zip(chunk_ids, labels)]

    if provider is not None and hasattr(provider, "update_cluster_assignments"):
        provider.update_cluster_assignments(updates)
        with resolve_connection(provider) as conn:
            _write_centroids(conn, centroids, cluster_sizes, tn)
            _write_cluster_sizes(conn, chunk_ids, labels, cluster_sizes, tn)
    else:
        conn = db_or_conn
        _write_centroids(conn, centroids, cluster_sizes, tn)
        _write_cluster_assignments(conn, chunk_ids, labels, cluster_sizes, tn)

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
    db_or_conn: Any,
    top_n: int = 5,
    table_names: TableNames | None = None,
) -> List[int]:
    """Find the *top_n* clusters closest to the query embedding.

    Uses cosine similarity against stored centroids.

    Args:
        query_embedding: Query vector.
        db_or_conn: Vector provider or DB connection.
        top_n: Number of clusters to return.

    Returns:
        Sorted list of cluster IDs (most similar first).
    """
    if hasattr(db_or_conn, "name"):
        try:
            with resolve_connection(db_or_conn) as conn:
                return _get_top_clusters_from_conn(query_embedding, conn, top_n, table_names)
        except Exception:
            return []
    return _get_top_clusters_from_conn(query_embedding, db_or_conn, top_n, table_names)


def _write_centroids(
    conn: Any,
    centroids: List[List[float]],
    cluster_sizes: Dict[int, int],
    table_names: TableNames,
) -> None:
    _sqlite = is_sqlite(conn)
    execute(conn, f"DELETE FROM {table_names.cluster_centroids}")
    for cluster_id, centroid in enumerate(centroids):
        if _sqlite:
            centroid_value = pack(f"{len(centroid)}f", *centroid)
        else:
            centroid_value = centroid
        execute(
            conn,
            f"INSERT INTO {table_names.cluster_centroids} (cluster_id, centroid, size) VALUES (?, ?, ?)",
            (cluster_id, centroid_value, cluster_sizes.get(cluster_id, 0)),
        )
    commit(conn)
    logger.info("Cluster build: %d centroids written.", len(centroids))


def _write_cluster_assignments(
    conn: Any,
    chunk_ids: List[str],
    labels: List[int],
    cluster_sizes: Dict[int, int],
    table_names: TableNames,
) -> None:
    batch_size = 1000
    n_chunks = len(chunk_ids)
    for i, (chunk_id, label) in enumerate(zip(chunk_ids, labels), 1):
        execute(
            conn,
            f"UPDATE {table_names.chunks} SET cluster_id = ?, cluster_size = ? WHERE chunk_id = ?",
            (int(label), cluster_sizes.get(label, 0), chunk_id),
        )
        if i % batch_size == 0 or i == n_chunks:
            commit(conn)
            logger.info("Cluster build: chunk assignments %s/%s written.", f"{i:,}", f"{n_chunks:,}")

    if table_exists(conn, table_names.vec_chunks):
        for i, (chunk_id, label) in enumerate(zip(chunk_ids, labels), 1):
            execute(
                conn,
                f"UPDATE {table_names.vec_chunks} SET cluster_id = ? WHERE chunk_id = ?",
                (int(label), chunk_id),
            )
            if i % batch_size == 0 or i == n_chunks:
                commit(conn)
                logger.info("Cluster build: vec assignments %s/%s written.", f"{i:,}", f"{n_chunks:,}")


def _write_cluster_sizes(
    conn: Any,
    chunk_ids: List[str],
    labels: List[int],
    cluster_sizes: Dict[int, int],
    table_names: TableNames,
) -> None:
    batch_size = 1000
    n_chunks = len(chunk_ids)
    for i, (chunk_id, label) in enumerate(zip(chunk_ids, labels), 1):
        execute(
            conn,
            f"UPDATE {table_names.chunks} SET cluster_size = ? WHERE chunk_id = ?",
            (cluster_sizes.get(label, 0), chunk_id),
        )
        if i % batch_size == 0 or i == n_chunks:
            commit(conn)


def _get_top_clusters_from_conn(
    query_embedding: List[float],
    conn: Any,
    top_n: int,
    table_names: TableNames | None = None,
) -> List[int]:
    tn = table_names or DEFAULT_TABLE_NAMES
    rows = fetchall(
        conn,
        f"SELECT cluster_id, centroid FROM {tn.cluster_centroids}",
    )

    if not rows:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    _sqlite = is_sqlite(conn)
    scores: List[Tuple[int, float]] = []
    for row in rows:
        cluster_id = int(row[0])
        raw = row[1]
        if _sqlite:
            blob = bytes(raw)
            dim = len(blob) // 4
            centroid = np.array(unpack(f"{dim}f", blob), dtype=np.float32)
        else:
            centroid = np.array(raw if isinstance(raw, list) else list(raw), dtype=np.float32)
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
