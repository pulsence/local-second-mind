"""Post-migration chunk enrichment pipeline.

Brings existing chunks up to date with the current ingest pipeline by
backfilling missing metadata in three tiers:

- **Tier 1** (in-place SQL): simhash, version/is_current/node_type defaults, tags.
- **Tier 2** (source-file re-parse): heading_path, start_char/end_char/chunk_length, graph.
- **Tier 2b** (embedding-only): cluster_id/cluster_size rebuild from existing embeddings.
- **Tier 3** (advisory): chunk-boundary drift and missing summary nodes — advise re-ingest.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lsm.db.tables import DEFAULT_TABLE_NAMES, TableNames

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Report dataclass
# ------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentReport:
    """Structured result from the enrichment pipeline."""

    tier1_updated: int = 0
    tier2_updated: int = 0
    tier2b_updated: int = 0
    tier2_skipped: tuple[str, ...] = ()
    tier3_needed: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


# ------------------------------------------------------------------
# Stale-chunk detection
# ------------------------------------------------------------------


def detect_stale_chunks(
    conn: sqlite3.Connection,
    table_names: TableNames | None = None,
    *,
    cluster_enabled: bool = False,
    summaries_enabled: bool = False,
) -> Dict[str, Any]:
    """Query the database to determine what enrichment is needed.

    Returns a dict keyed by tier with counts and lists.
    """
    tn = table_names or DEFAULT_TABLE_NAMES

    # Check if chunks table exists
    exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (tn.chunks,),
    ).fetchone()[0]
    if not exists:
        return {
            "tier1": {
                "simhash_null_count": 0,
                "version_null_count": 0,
                "node_type_null_count": 0,
                "tags_missing_count": 0,
            },
            "tier2": {
                "heading_path_null_count": 0,
                "positions_null_count": 0,
                "source_paths": [],
                "cluster_rebuild_needed": False,
            },
            "tier3": {
                "schema_diff": {},
                "missing_section_summary_files": [],
                "missing_file_summary_files": [],
                "needs_reingest": False,
            },
        }

    # --- Tier 1 counts ---
    simhash_null = conn.execute(
        f"SELECT COUNT(*) FROM {tn.chunks} WHERE simhash IS NULL AND is_current = 1"
    ).fetchone()[0]
    version_null = conn.execute(
        f"SELECT COUNT(*) FROM {tn.chunks} WHERE version IS NULL"
    ).fetchone()[0]
    node_type_null = conn.execute(
        f"SELECT COUNT(*) FROM {tn.chunks} WHERE node_type IS NULL OR node_type = ''"
    ).fetchone()[0]
    tags_missing = conn.execute(
        f"SELECT COUNT(*) FROM {tn.chunks} WHERE (root_tags IS NULL OR content_type IS NULL) AND is_current = 1"
    ).fetchone()[0]

    # --- Tier 2 counts ---
    heading_path_null = conn.execute(
        f"""SELECT COUNT(*) FROM {tn.chunks}
            WHERE heading IS NOT NULL AND heading_path IS NULL
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1"""
    ).fetchone()[0]
    positions_null = conn.execute(
        f"""SELECT COUNT(*) FROM {tn.chunks}
            WHERE start_char IS NULL
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1"""
    ).fetchone()[0]

    # Distinct source paths needing tier 2 enrichment
    source_rows = conn.execute(
        f"""SELECT DISTINCT source_path FROM {tn.chunks}
            WHERE (
                (heading IS NOT NULL AND heading_path IS NULL)
                OR start_char IS NULL
            )
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1"""
    ).fetchall()
    source_paths = [row[0] for row in source_rows]

    # Cluster rebuild needed?
    cluster_rebuild = False
    if cluster_enabled:
        null_cluster = conn.execute(
            f"SELECT COUNT(*) FROM {tn.chunks} WHERE cluster_id IS NULL AND is_current = 1"
        ).fetchone()[0]
        cluster_rebuild = null_cluster > 0

    # --- Tier 3 ---
    missing_section_files: List[str] = []
    missing_file_files: List[str] = []
    if summaries_enabled:
        # Files that have chunk rows but no section_summary rows
        section_rows = conn.execute(
            f"""SELECT DISTINCT c.source_path FROM {tn.chunks} c
                WHERE c.is_current = 1
                AND c.node_type = 'chunk'
                AND NOT EXISTS (
                    SELECT 1 FROM {tn.chunks} s
                    WHERE s.source_path = c.source_path
                    AND s.node_type = 'section_summary'
                    AND s.is_current = 1
                )"""
        ).fetchall()
        missing_section_files = [r[0] for r in section_rows]

        file_rows = conn.execute(
            f"""SELECT DISTINCT c.source_path FROM {tn.chunks} c
                WHERE c.is_current = 1
                AND c.node_type = 'chunk'
                AND NOT EXISTS (
                    SELECT 1 FROM {tn.chunks} s
                    WHERE s.source_path = c.source_path
                    AND s.node_type = 'file_summary'
                    AND s.is_current = 1
                )"""
        ).fetchall()
        missing_file_files = [r[0] for r in file_rows]

    needs_reingest = bool(missing_section_files or missing_file_files)

    return {
        "tier1": {
            "simhash_null_count": simhash_null,
            "version_null_count": version_null,
            "node_type_null_count": node_type_null,
            "tags_missing_count": tags_missing,
        },
        "tier2": {
            "heading_path_null_count": heading_path_null,
            "positions_null_count": positions_null,
            "source_paths": source_paths,
            "cluster_rebuild_needed": cluster_rebuild,
        },
        "tier3": {
            "schema_diff": {},
            "missing_section_summary_files": missing_section_files,
            "missing_file_summary_files": missing_file_files,
            "needs_reingest": needs_reingest,
        },
    }


# ------------------------------------------------------------------
# Tier 1: in-place SQL updates
# ------------------------------------------------------------------


def run_tier1_enrichment(
    conn: sqlite3.Connection,
    config: Any,
    table_names: TableNames | None = None,
) -> int:
    """Run Tier 1 enrichment: simhash, defaults, tags.

    Returns count of rows updated.
    """
    tn = table_names or DEFAULT_TABLE_NAMES
    total = 0

    # 1. Simhash backfill
    total += _backfill_simhash(conn, tn)

    # 2. Version/is_current defaults
    total += _backfill_defaults(conn, tn)

    # 3. Node-type defaults + vec table sync
    total += _backfill_node_type(conn, tn)

    # 4. Tag enrichment from config roots
    total += _backfill_tags(conn, config, tn)

    conn.commit()
    return total


def _backfill_simhash(conn: sqlite3.Connection, tn: TableNames) -> int:
    """Compute and store simhash for chunks where it is NULL."""
    from lsm.ingest.dedup_hash import compute_simhash

    batch_size = 1000
    updated = 0

    while True:
        rows = conn.execute(
            f"SELECT chunk_id, chunk_text FROM {tn.chunks} WHERE simhash IS NULL AND is_current = 1 LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        for chunk_id, chunk_text in rows:
            h = compute_simhash(chunk_text or "")
            conn.execute(
                f"UPDATE {tn.chunks} SET simhash = ? WHERE chunk_id = ?",
                (h, chunk_id),
            )
            updated += 1
        logger.info("Simhash backfill: %d chunks updated so far", updated)

    return updated


def _backfill_defaults(conn: sqlite3.Connection, tn: TableNames) -> int:
    """Set version and is_current defaults where NULL."""
    updated = 0

    cur = conn.execute(
        f"UPDATE {tn.chunks} SET version = 1 WHERE version IS NULL"
    )
    updated += cur.rowcount

    cur = conn.execute(
        f"UPDATE {tn.chunks} SET is_current = 1 WHERE is_current IS NULL"
    )
    updated += cur.rowcount

    return updated


def _backfill_node_type(conn: sqlite3.Connection, tn: TableNames) -> int:
    """Set node_type to 'chunk' where NULL or empty, and sync to vec table."""
    cur = conn.execute(
        f"UPDATE {tn.chunks} SET node_type = 'chunk' WHERE node_type IS NULL OR node_type = ''"
    )
    updated = cur.rowcount

    # Mirror to vec_chunks if the table exists
    vec_exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (tn.vec_chunks,),
    ).fetchone()[0]
    if vec_exists:
        conn.execute(
            f"""UPDATE {tn.vec_chunks} SET node_type = 'chunk'
                WHERE chunk_id IN (
                    SELECT chunk_id FROM {tn.chunks}
                    WHERE node_type = 'chunk'
                )
                AND (node_type IS NULL OR node_type = '')"""
        )
        # Sync is_current too
        conn.execute(
            f"""UPDATE {tn.vec_chunks} SET is_current = 1
                WHERE chunk_id IN (
                    SELECT chunk_id FROM {tn.chunks} WHERE is_current = 1
                )
                AND (is_current IS NULL OR is_current = 0)"""
        )

    return updated


def _backfill_tags(
    conn: sqlite3.Connection,
    config: Any,
    tn: TableNames,
) -> int:
    """Apply root_tags and content_type from config to chunks missing them."""
    roots = getattr(getattr(config, "ingest", None), "roots", None)
    if not roots:
        return 0

    from lsm.ingest.fs import collect_folder_tags

    updated = 0

    # Build a mapping of resolved root paths to their config
    root_map: List[Tuple[Path, Any]] = []
    for root_cfg in roots:
        resolved = root_cfg.path.expanduser().resolve()
        root_map.append((resolved, root_cfg))

    # Fetch chunks needing tags
    rows = conn.execute(
        f"""SELECT chunk_id, source_path FROM {tn.chunks}
            WHERE (root_tags IS NULL OR content_type IS NULL)
            AND is_current = 1"""
    ).fetchall()

    for chunk_id, source_path in rows:
        sp = Path(source_path)
        matched_root = None
        matched_cfg = None
        for root_path, root_cfg in root_map:
            try:
                sp.resolve().relative_to(root_path)
                matched_root = root_path
                matched_cfg = root_cfg
                break
            except ValueError:
                continue

        if matched_cfg is None:
            continue

        root_tags = json.dumps(matched_cfg.tags or [])
        content_type = matched_cfg.content_type or ""
        folder_tags = json.dumps(collect_folder_tags(sp, matched_root))

        conn.execute(
            f"""UPDATE {tn.chunks}
                SET root_tags = COALESCE(root_tags, ?),
                    folder_tags = COALESCE(folder_tags, ?),
                    content_type = COALESCE(content_type, ?)
                WHERE chunk_id = ?""",
            (root_tags, folder_tags, content_type, chunk_id),
        )
        updated += 1

    return updated


# ------------------------------------------------------------------
# Tier 2: source-file re-parse
# ------------------------------------------------------------------


def run_tier2_enrichment(
    conn: sqlite3.Connection,
    config: Any,
    table_names: TableNames | None = None,
    root_paths: list[Path] | None = None,
) -> Tuple[int, List[str]]:
    """Run Tier 2 enrichment: heading_path, positions, graph backfill.

    Args:
        conn: SQLite connection.
        config: LSMConfig instance.
        table_names: Table name registry.
        root_paths: List of root directories to search for source files.

    Returns:
        Tuple of (updated_count, skipped_source_paths).
    """
    tn = table_names or DEFAULT_TABLE_NAMES
    updated = 0
    skipped: List[str] = []

    # Get distinct source paths needing tier 2 enrichment
    rows = conn.execute(
        f"""SELECT DISTINCT source_path FROM {tn.chunks}
            WHERE (
                (heading IS NOT NULL AND heading_path IS NULL)
                OR start_char IS NULL
            )
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1"""
    ).fetchall()

    for (source_path,) in rows:
        sp = Path(source_path)
        if not sp.exists():
            skipped.append(source_path)
            continue

        # Heading path backfill
        updated += _backfill_heading_path(conn, tn, source_path, sp)

        # Position backfill
        updated += _backfill_positions(conn, tn, source_path, sp)

    # Graph backfill for files without graph entries
    updated += _backfill_graph(conn, tn)

    conn.commit()
    return updated, skipped


def _backfill_heading_path(
    conn: sqlite3.Connection,
    tn: TableNames,
    source_path: str,
    file_path: Path,
) -> int:
    """Re-parse a file and backfill heading_path for chunks that need it."""
    rows = conn.execute(
        f"""SELECT chunk_id, heading, chunk_index FROM {tn.chunks}
            WHERE source_path = ?
            AND heading IS NOT NULL AND heading_path IS NULL
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1""",
        (source_path,),
    ).fetchall()

    if not rows:
        return 0

    try:
        from lsm.utils.file_graph import build_file_graph

        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        fg = build_file_graph(file_path, raw_text)

        # Build heading path map from graph nodes
        heading_map = _build_heading_map_from_graph(fg)
    except Exception as exc:
        logger.warning("Cannot re-parse %s for heading_path: %s", source_path, exc)
        return 0

    updated = 0
    for chunk_id, heading, chunk_index in rows:
        # Try to find heading path by matching heading text
        path = heading_map.get(heading)
        if path is not None:
            conn.execute(
                f"UPDATE {tn.chunks} SET heading_path = ? WHERE chunk_id = ?",
                (json.dumps(path), chunk_id),
            )
            updated += 1

    return updated


def _build_heading_map_from_graph(fg: Any) -> Dict[str, List[str]]:
    """Build a mapping of heading text → heading path list from a FileGraph."""
    heading_map: Dict[str, List[str]] = {}
    if not hasattr(fg, "nodes"):
        return heading_map

    # Build parent lookup
    node_by_id: Dict[Any, Any] = {}
    for node in fg.nodes:
        node_by_id[node.node_id] = node

    for node in fg.nodes:
        if node.node_type == "heading":
            label = node.label or ""
            # Walk parent chain to build path
            path: List[str] = [label]
            current = node
            while current.parent_id is not None and current.parent_id in node_by_id:
                parent = node_by_id[current.parent_id]
                if parent.node_type == "heading":
                    path.insert(0, parent.label or "")
                current = parent
            heading_map[label] = path

    return heading_map


def _backfill_positions(
    conn: sqlite3.Connection,
    tn: TableNames,
    source_path: str,
    file_path: Path,
) -> int:
    """Backfill start_char/end_char/chunk_length from source file re-parse."""
    rows = conn.execute(
        f"""SELECT chunk_id, chunk_text, chunk_index FROM {tn.chunks}
            WHERE source_path = ?
            AND start_char IS NULL
            AND (node_type = 'chunk' OR node_type IS NULL)
            AND is_current = 1
            ORDER BY chunk_index""",
        (source_path,),
    ).fetchall()

    if not rows:
        return 0

    try:
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Cannot read %s for position backfill: %s", source_path, exc)
        return 0

    updated = 0
    for chunk_id, chunk_text, chunk_index in rows:
        if not chunk_text:
            continue
        # Try to locate the chunk text in the source file
        prefix = chunk_text[:100]
        idx = raw_text.find(prefix)
        if idx >= 0:
            start_char = idx
            end_char = idx + len(chunk_text)
            chunk_length = len(chunk_text)
            conn.execute(
                f"""UPDATE {tn.chunks}
                    SET start_char = ?, end_char = ?, chunk_length = ?
                    WHERE chunk_id = ?""",
                (start_char, end_char, chunk_length, chunk_id),
            )
            updated += 1
        else:
            logger.debug(
                "Chunk text prefix mismatch for %s chunk_index=%s (boundary drift → Tier 3)",
                source_path,
                chunk_index,
            )

    return updated


def _backfill_graph(conn: sqlite3.Connection, tn: TableNames) -> int:
    """Create graph nodes/edges for source files without graph entries."""
    # Check if graph tables exist
    graph_exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (tn.graph_nodes,),
    ).fetchone()[0]
    if not graph_exists:
        return 0

    # Find source files that have chunks but no graph nodes
    rows = conn.execute(
        f"""SELECT DISTINCT c.source_path FROM {tn.chunks} c
            WHERE c.is_current = 1
            AND c.node_type = 'chunk'
            AND NOT EXISTS (
                SELECT 1 FROM {tn.graph_nodes} g
                WHERE g.source_path = c.source_path
            )"""
    ).fetchall()

    updated = 0
    for (source_path,) in rows:
        sp = Path(source_path)
        if not sp.exists():
            continue

        try:
            from lsm.utils.file_graph import build_file_graph

            raw_text = sp.read_text(encoding="utf-8", errors="replace")
            fg = build_file_graph(sp, raw_text)

            for node in fg.nodes:
                heading_path_val = None
                if hasattr(node, "heading_path") and node.heading_path:
                    heading_path_val = json.dumps(node.heading_path)

                conn.execute(
                    f"""INSERT OR IGNORE INTO {tn.graph_nodes}
                        (node_id, node_type, label, source_path, heading_path)
                        VALUES (?, ?, ?, ?, ?)""",
                    (
                        node.node_id,
                        node.node_type,
                        node.label or "",
                        source_path,
                        heading_path_val,
                    ),
                )

            if hasattr(fg, "edges"):
                for edge in fg.edges:
                    conn.execute(
                        f"""INSERT OR IGNORE INTO {tn.graph_edges}
                            (src_id, dst_id, edge_type, weight)
                            VALUES (?, ?, ?, ?)""",
                        (
                            edge.src_id,
                            edge.dst_id,
                            getattr(edge, "edge_type", "contains"),
                            getattr(edge, "weight", 1.0),
                        ),
                    )

            updated += 1
        except Exception as exc:
            logger.warning("Cannot build graph for %s: %s", source_path, exc)

    return updated


# ------------------------------------------------------------------
# Tier 2b: cluster rebuild
# ------------------------------------------------------------------


def run_tier2_cluster_enrichment(
    conn: sqlite3.Connection,
    config: Any,
    table_names: TableNames | None = None,
) -> int:
    """Rebuild cluster assignments from existing embeddings.

    Returns count of chunks updated.
    """
    tn = table_names or DEFAULT_TABLE_NAMES

    query_cfg = getattr(config, "query", None)
    if query_cfg is None or not getattr(query_cfg, "cluster_enabled", False):
        return 0

    # Check if vec_chunks exists
    vec_exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (tn.vec_chunks,),
    ).fetchone()[0]
    if not vec_exists:
        return 0

    null_cluster = conn.execute(
        f"SELECT COUNT(*) FROM {tn.chunks} WHERE cluster_id IS NULL AND is_current = 1"
    ).fetchone()[0]
    if null_cluster == 0:
        return 0

    from lsm.db.clustering import build_clusters

    algorithm = getattr(query_cfg, "cluster_algorithm", "kmeans")
    k = getattr(query_cfg, "cluster_k", 50)

    result = build_clusters(conn, algorithm=algorithm, k=k, table_names=tn)
    return result.get("n_chunks", 0)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------


def run_enrichment_pipeline(
    conn: sqlite3.Connection,
    config: Any,
    table_names: TableNames | None = None,
    root_paths: list[Path] | None = None,
    skip_tier2: bool = False,
    stage_tracker: Optional[Callable[[str, str], None]] = None,
    skip_stages: Optional[set[str]] = None,
) -> EnrichmentReport:
    """Orchestrate full enrichment pipeline.

    Args:
        conn: SQLite connection.
        config: LSMConfig instance.
        table_names: Table name registry.
        root_paths: Root paths for source file lookups.
        skip_tier2: Skip Tier 2 enrichment.
        stage_tracker: Optional callback(stage_name, status) for progress tracking.
        skip_stages: Optional stage names to skip (resume support).

    Returns:
        EnrichmentReport with all results.
    """
    tn = table_names or DEFAULT_TABLE_NAMES
    errors: List[str] = []
    failed_stages: set[str] = set()
    skipped_stage_names = set(skip_stages or set())

    cluster_enabled = getattr(
        getattr(config, "query", None), "cluster_enabled", False
    )
    summaries_enabled = getattr(
        getattr(config, "query", None), "retrieval_profile", ""
    ) == "multi_vector"

    # Detect what's needed
    stale = detect_stale_chunks(
        conn, tn,
        cluster_enabled=cluster_enabled,
        summaries_enabled=summaries_enabled,
    )

    def _run_stage(stage_name: str, fn: Callable[[], int]) -> int:
        if stage_name in skipped_stage_names:
            logger.info("Skipping enrichment stage %s (already completed)", stage_name)
            return 0
        try:
            if stage_tracker:
                stage_tracker(stage_name, "in_progress")
            updated = int(fn())
            if stage_tracker:
                stage_tracker(stage_name, "completed")
            return updated
        except Exception as exc:
            errors.append(f"{stage_name} error: {exc}")
            failed_stages.add(stage_name)
            logger.error("%s failed: %s", stage_name, exc)
            if stage_tracker:
                stage_tracker(stage_name, "failed")
            return 0

    # Tier 1
    tier1_updated = 0
    if stage_tracker:
        stage_tracker("enrich_tier1", "in_progress")
    tier1_updated += _run_stage("enrich_tier1_simhash", lambda: _backfill_simhash(conn, tn))
    tier1_updated += _run_stage("enrich_tier1_defaults", lambda: _backfill_defaults(conn, tn))
    tier1_updated += _run_stage("enrich_tier1_node_type", lambda: _backfill_node_type(conn, tn))
    tier1_updated += _run_stage("enrich_tier1_tags", lambda: _backfill_tags(conn, config, tn))
    if stage_tracker:
        tier1_failed = any(
            stage_name in failed_stages
            for stage_name in (
                "enrich_tier1_simhash",
                "enrich_tier1_defaults",
                "enrich_tier1_node_type",
                "enrich_tier1_tags",
            )
        )
        stage_tracker("enrich_tier1", "failed" if tier1_failed else "completed")
    conn.commit()
    logger.info("Tier 1 enrichment: %d chunks updated", tier1_updated)

    # Tier 2
    tier2_updated = 0
    tier2_skipped: List[str] = []
    if not skip_tier2:
        if stage_tracker:
            stage_tracker("enrich_tier2", "in_progress")
        source_rows = conn.execute(
            f"""SELECT DISTINCT source_path FROM {tn.chunks}
                WHERE (
                    (heading IS NOT NULL AND heading_path IS NULL)
                    OR start_char IS NULL
                )
                AND (node_type = 'chunk' OR node_type IS NULL)
                AND is_current = 1"""
        ).fetchall()
        source_paths = [str(row[0]) for row in source_rows]

        existing_paths: List[tuple[str, Path]] = []
        for source_path in source_paths:
            candidate = Path(source_path)
            if candidate.exists():
                existing_paths.append((source_path, candidate))
            else:
                tier2_skipped.append(source_path)

        def _run_heading_path() -> int:
            updated = 0
            for source_path, file_path in existing_paths:
                updated += _backfill_heading_path(conn, tn, source_path, file_path)
            return updated

        def _run_positions() -> int:
            updated = 0
            for source_path, file_path in existing_paths:
                updated += _backfill_positions(conn, tn, source_path, file_path)
            return updated

        tier2_updated += _run_stage("enrich_tier2_heading_path", _run_heading_path)
        tier2_updated += _run_stage("enrich_tier2_positions", _run_positions)
        tier2_updated += _run_stage("enrich_tier2_graph", lambda: _backfill_graph(conn, tn))
        conn.commit()
        logger.info(
            "Tier 2 enrichment: %d updated, %d skipped",
            tier2_updated,
            len(tier2_skipped),
        )
        if stage_tracker:
            tier2_failed = any(
                stage_name in failed_stages
                for stage_name in (
                    "enrich_tier2_heading_path",
                    "enrich_tier2_positions",
                    "enrich_tier2_graph",
                )
            )
            stage_tracker("enrich_tier2", "failed" if tier2_failed else "completed")

    # Tier 2b cluster rebuild
    tier2b_updated = 0
    if stale["tier2"]["cluster_rebuild_needed"]:
        if stage_tracker:
            stage_tracker("enrich_tier2b", "in_progress")
        tier2b_updated = _run_stage(
            "enrich_tier2_clusters",
            lambda: run_tier2_cluster_enrichment(conn, config, tn),
        )
        if stage_tracker:
            stage_tracker(
                "enrich_tier2b",
                "failed" if "enrich_tier2_clusters" in failed_stages else "completed",
            )
        logger.info("Tier 2b cluster rebuild: %d chunks", tier2b_updated)

    # Tier 3 advisory
    tier3_needed: List[str] = []
    if "enrich_tier3_gap_detection" in skipped_stage_names:
        logger.info("Skipping enrichment stage enrich_tier3_gap_detection (already completed)")
    else:
        if stage_tracker:
            stage_tracker("enrich_tier3_gap_detection", "in_progress")
        if stale["tier3"]["needs_reingest"]:
            for f in stale["tier3"]["missing_section_summary_files"]:
                tier3_needed.append(f"missing section summary: {f}")
            for f in stale["tier3"]["missing_file_summary_files"]:
                tier3_needed.append(f"missing file summary: {f}")
        if stage_tracker:
            stage_tracker("enrich_tier3_gap_detection", "completed")

    return EnrichmentReport(
        tier1_updated=tier1_updated,
        tier2_updated=tier2_updated,
        tier2b_updated=tier2b_updated,
        tier2_skipped=tuple(tier2_skipped),
        tier3_needed=tuple(tier3_needed),
        errors=tuple(errors),
    )
