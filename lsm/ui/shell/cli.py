"""
CLI entry point for ingest single-shot commands.

Provides build, tag, and wipe command runners for non-interactive usage.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from lsm.config import load_config_from_file
from lsm.config.models import LSMConfig, DBConfig
from lsm.db.tables import TableNames
from lsm.logging import get_logger
from lsm.db.migration import migrate as migrate_db
from lsm.ingest.api import run_ingest as api_run_ingest, wipe_collection as api_wipe_collection
from lsm.ingest.tagging import tag_chunks
from lsm.vectordb import PruneCriteria, create_vectordb_provider

logger = get_logger(__name__)


def _table_names(config: LSMConfig) -> TableNames:
    return TableNames(prefix=config.db.table_prefix)


def run_ingest(args) -> int:
    """Run ingest command with build/tag/wipe subcommands."""
    command = getattr(args, "ingest_command", None)
    if command == "build":
        return run_build_cli(
            args.config,
            force=getattr(args, "force", False),
            skip_errors=getattr(args, "skip_errors", None),
            dry_run=getattr(args, "dry_run", None),
            force_reingest_changed_config=getattr(args, "force_reingest_changed_config", False),
            force_file_pattern=getattr(args, "force_file_pattern", None),
        )
    if command == "tag":
        return run_tag_cli(args.config, max_chunks=getattr(args, "max", None))
    if command == "wipe":
        return run_wipe_cli(args.config, confirm=getattr(args, "confirm", False))

    print("Missing ingest subcommand. Use `lsm ingest --help` for options.")
    return 2


def run_db(args) -> int:
    """Run db maintenance command."""
    command = getattr(args, "db_command", None)
    if command == "prune":
        return run_db_prune_cli(
            args.config,
            max_versions=getattr(args, "max_versions", None),
            older_than_days=getattr(args, "older_than_days", None),
        )
    if command == "complete":
        return run_db_complete_cli(
            args.config,
            force_file_pattern=getattr(args, "force_file_pattern", None),
        )

    print("Missing db subcommand. Use `lsm db --help` for options.")
    return 2


def run_migrate(args) -> int:
    """Run explicit migration command."""
    return run_migrate_cli(
        args.config,
        from_db=getattr(args, "from_db", None),
        to_db=getattr(args, "to_db", None),
        from_version=getattr(args, "from_version", None),
        to_version=getattr(args, "to_version", None),
        resume=getattr(args, "resume", False),
        enrich=getattr(args, "enrich", False),
        skip_enrich=getattr(args, "skip_enrich", False),
        rechunk=getattr(args, "rechunk", False),
        skip_rechunk=getattr(args, "skip_rechunk", False),
        stages=getattr(args, "stages", None),
        source_path=getattr(args, "source_path", None),
        source_collection=getattr(args, "source_collection", None),
        source_connection_string=getattr(args, "source_connection_string", None),
        source_dir=getattr(args, "source_dir", None),
        target_path=getattr(args, "target_path", None),
        target_collection=getattr(args, "target_collection", None),
        target_connection_string=getattr(args, "target_connection_string", None),
        batch_size=getattr(args, "batch_size", 1000),
    )


def run_cache(args, config: LSMConfig) -> int:
    """Run cache maintenance commands."""
    command = getattr(args, "cache_command", None)
    if command == "clear":
        return run_cache_clear_cli(
            config,
            clear_reranker=bool(getattr(args, "reranker", False)),
        )

    print("Missing cache subcommand. Use `lsm cache --help` for options.")
    return 2


def _load_config(config_path: str | Path) -> LSMConfig:
    """Load and validate configuration."""
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        raise FileNotFoundError(
            f"Ingest config not found: {cfg_path}\n"
            f"Either create it or pass --config explicitly."
        )

    logger.info(f"Loading configuration from: {cfg_path}")
    return load_config_from_file(cfg_path)


def _print_post_ingest_advisories(config: LSMConfig) -> None:
    """Check and print post-ingest advisories on CLI path."""
    try:
        from lsm.db.job_status import check_job_advisories

        provider = create_vectordb_provider(config.db)
        conn = getattr(provider, "connection", None)
        if conn is not None:
            advisories = check_job_advisories(conn, config)
        else:
            get_conn = getattr(provider, "_get_conn", None)
            if get_conn is None:
                return
            with get_conn() as pg_conn:
                advisories = check_job_advisories(pg_conn, config)

        for adv in advisories:
            print(f"\n[advisory] {adv.message}")
            print(f"  Run: {adv.action}")
    except Exception:
        logger.debug("Post-ingest advisory check failed", exc_info=True)


def run_build_cli(
    config_path: str | Path,
    force: bool = False,
    skip_errors: Optional[bool] = None,
    dry_run: Optional[bool] = None,
    force_reingest_changed_config: bool = False,
    force_file_pattern: Optional[str] = None,
) -> int:
    """
    Run the ingest build pipeline.

    Args:
        config_path: Path to configuration file
        force: If True, clears manifest to re-ingest all files
        skip_errors: Override skip_errors config
        dry_run: Override dry_run config

    Returns:
        Exit code (0 for success)
    """
    config = _load_config(config_path)

    if skip_errors is not None:
        config.ingest.skip_errors = skip_errors
    if dry_run is not None:
        config.ingest.dry_run = dry_run

    def progress(event: str, current: int, total: int, message: str) -> None:
        if total > 0:
            print(f"[{event}] {current}/{total} {message}")
        else:
            print(f"[{event}] {message}")

    api_run_ingest(
        config,
        force=force,
        force_reingest_changed_config=force_reingest_changed_config,
        force_file_pattern=force_file_pattern,
        progress_callback=progress,
    )

    logger.info("Ingest completed successfully")

    # Post-ingest advisories
    _print_post_ingest_advisories(config)

    return 0


def run_tag_cli(
    config_path: str | Path,
    max_chunks: Optional[int] = None,
) -> int:
    """
    Run AI tagging on untagged chunks.

    Args:
        config_path: Path to configuration file
        max_chunks: Optional limit on number of chunks to tag

    Returns:
        Exit code (0 for success)
    """
    config = _load_config(config_path)
    provider = create_vectordb_provider(config.db)
    tagging_config = config.llm.get_tagging_config()

    print("\nStarting AI tagging...")
    print(f"Using model: {tagging_config.model}")
    print(f"Provider: {tagging_config.provider}")
    if max_chunks:
        print(f"Max chunks to tag: {max_chunks}")

    tagged, failed = tag_chunks(
        collection=provider,
        llm_config=tagging_config,
        num_tags=3,
        batch_size=100,
        max_chunks=max_chunks,
        dry_run=False,
    )

    print("\nTagging complete")
    print(f"Successfully tagged: {tagged} chunks")
    print(f"Failed: {failed} chunks")

    return 0


def run_wipe_cli(
    config_path: str | Path,
    confirm: bool = False,
) -> int:
    """
    Wipe the vector DB collection.

    Args:
        config_path: Path to configuration file
        confirm: Require explicit confirmation

    Returns:
        Exit code (0 for success)
    """
    if not confirm:
        print("Refusing to wipe without confirmation. Use --confirm to proceed.")
        return 2

    config = _load_config(config_path)
    try:
        deleted = api_wipe_collection(config)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"\nDeleted {deleted:,} chunks from collection '{config.collection}'.")
    print("Collection cleared successfully.")
    return 0


def run_db_prune_cli(
    config_path: str | Path,
    *,
    max_versions: Optional[int] = None,
    older_than_days: Optional[int] = None,
) -> int:
    """Run non-current version prune operation."""
    config = _load_config(config_path)
    provider = create_vectordb_provider(config.db)

    try:
        deleted = provider.prune_old_versions(
            PruneCriteria(
                max_versions=max_versions,
                older_than_days=older_than_days,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(
        "Prune complete: "
        f"deleted {deleted:,} non-current chunks "
        f"(max_versions={max_versions}, older_than_days={older_than_days})."
    )
    return 0


def run_db_complete_cli(
    config_path: str | Path,
    *,
    force_file_pattern: Optional[str] = None,
) -> int:
    """Run selective completion ingest for changed config state."""
    config = _load_config(config_path)

    def progress(event: str, current: int, total: int, message: str) -> None:
        if total > 0:
            print(f"[{event}] {current}/{total} {message}")
        else:
            print(f"[{event}] {message}")

    try:
        api_run_ingest(
            config,
            force=False,
            force_reingest_changed_config=True,
            force_file_pattern=force_file_pattern,
            progress_callback=progress,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Completion ingest finished successfully.")
    return 0


def _with_overrides(
    base: DBConfig,
    *,
    provider: Optional[str] = None,
    path: Optional[str] = None,
    collection: Optional[str] = None,
    connection_string: Optional[str] = None,
) -> DBConfig:
    config = replace(base)
    if provider:
        config.provider = str(provider)
    if path:
        config.path = Path(path).expanduser()
    if collection:
        config.collection = str(collection)
    if connection_string:
        config.connection_string = str(connection_string)
    return config


def run_migrate_cli(
    config_path: str | Path,
    *,
    from_db: Optional[str] = None,
    to_db: Optional[str] = None,
    from_version: Optional[str] = None,
    to_version: Optional[str] = None,
    resume: bool = False,
    enrich: bool = False,
    skip_enrich: bool = False,
    rechunk: bool = False,
    skip_rechunk: bool = False,
    stages: Optional[list[str]] = None,
    source_path: Optional[str] = None,
    source_collection: Optional[str] = None,
    source_connection_string: Optional[str] = None,
    source_dir: Optional[str] = None,
    target_path: Optional[str] = None,
    target_collection: Optional[str] = None,
    target_connection_string: Optional[str] = None,
    batch_size: int = 1000,
) -> int:
    """Run backend migration command."""
    if enrich and skip_enrich:
        print("Error: --enrich and --skip-enrich are mutually exclusive.")
        return 2

    if rechunk and skip_rechunk:
        print("Error: --rechunk and --skip-rechunk are mutually exclusive.")
        return 2

    if enrich and (from_db or to_db):
        print("Error: --enrich is for in-place enrichment. Do not use with --from-db/--to-db.")
        return 2

    if stages and not enrich:
        print("Error: --stage requires --enrich.")
        return 2

    # Resolve stage names early so invalid names fail fast
    only_stages: Optional[set[str]] = None
    if stages:
        from lsm.db.enrichment import resolve_stage_names

        try:
            only_stages = resolve_stage_names(stages)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 2

    config = _load_config(config_path)

    # Standalone enrichment mode: enrich existing database, no backend copy
    if enrich:
        return _run_standalone_enrichment(
            config, rechunk=rechunk, skip_rechunk=skip_rechunk, only_stages=only_stages,
        )

    # Auto-detect when no explicit source specified
    if not from_db and not from_version:
        try:
            from lsm.db.migration import auto_detect_migration

            global_folder = getattr(config.global_settings, "global_folder", None)
            if global_folder:
                detected = auto_detect_migration(global_folder, config)
                from_db = detected.get("from_db")
                from_version = from_version or detected.get("from_version")
                to_db = to_db or detected.get("to_db")
                source_dir = source_dir or detected.get("source_dir")
                if from_db or from_version:
                    logger.info(
                        "Auto-detected migration: from_db=%s, from_version=%s, to_db=%s",
                        from_db, from_version, to_db,
                    )
        except Exception as exc:
            print(f"Auto-detection failed: {exc}")

    # Determine effective source and target values
    # Prefer from_db (backend copy) over from_version (legacy format migration)
    # so that e.g. chroma + v0.7 routes to the chroma vector-copy path.
    if from_db:
        source_value = from_db.strip().lower()
    elif from_version and from_version.startswith("v0.7"):
        source_value = "v0.7"
    else:
        print("Error: cannot determine migration source. Use --from-db or --from-version.")
        return 2

    target_value = (to_db or config.db.provider).strip().lower()

    target_vdb = _with_overrides(
        config.db,
        provider=target_value,
        path=target_path,
        collection=target_collection,
        connection_string=target_connection_string,
    )

    source_payload: object
    if source_value == "chroma":
        if source_path:
            chroma_path = source_path
        else:
            global_folder = getattr(config.global_settings, "global_folder", None)
            chroma_path = str(Path(global_folder) / ".chroma") if global_folder else ".chroma"
        source_payload = _with_overrides(
            config.db,
            provider="chromadb",
            path=chroma_path,
            collection=source_collection or config.db.collection,
            connection_string=source_connection_string,
        )
    elif source_value in {"sqlite", "postgresql"}:
        source_payload = _with_overrides(
            config.db,
            provider=source_value,
            path=source_path,
            collection=source_collection,
            connection_string=source_connection_string,
        )
    elif source_value == "v0.7":
        if target_value not in {"sqlite"}:
            print("Error: legacy v0.7 migration only supports --to-db sqlite.")
            return 2
        default_source_dir = source_dir
        if default_source_dir is None:
            default_source_dir = str(getattr(config.global_settings, "global_folder", "") or "")
        source_payload = {"source_dir": default_source_dir}
    else:
        print(f"Error: unsupported migration source '{source_value}'.")
        return 2

    target_runtime = replace(config, db=target_vdb)
    tn = _table_names(target_runtime)

    def progress(stage: str, current: int, total: int, message: str) -> None:
        logger.info("%s", message)

    try:
        result = migrate_db(
            source=source_value,
            target=target_value,
            source_config=source_payload,
            target_config=target_runtime,
            progress_callback=progress,
            batch_size=max(1, int(batch_size)),
            resume=resume,
            skip_enrich=skip_enrich,
            table_names=tn,
        )
    except KeyboardInterrupt:
        print("\nMigration interrupted. Progress has been saved.")
        print("Run `lsm migrate --resume` to continue from where you left off.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    if result.get("interrupted"):
        return 130

    print(
        "Migration complete: "
        f"{result.get('migrated_vectors', 0):,}/{result.get('total_vectors', 0):,} vectors, "
        f"validated tables={result.get('validated_tables', 0)}."
    )
    report = result.get("enrichment")
    if report is not None:
        _print_enrichment_summary(report)
        _handle_rechunk_offer(config, report, rechunk=rechunk, skip_rechunk=skip_rechunk)

    return 0


def _run_standalone_enrichment(
    config: LSMConfig,
    *,
    rechunk: bool = False,
    skip_rechunk: bool = False,
    only_stages: Optional[set[str]] = None,
) -> int:
    """Run enrichment on the existing database without backend migration."""
    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", getattr(provider, "_conn", None))
    if conn is None:
        print("Error: Enrichment requires a backend with direct SQL access.")
        return 1

    from lsm.db.enrichment import run_enrichment_pipeline

    print("Running standalone enrichment on existing database...")
    report = run_enrichment_pipeline(
        conn, config, table_names=_table_names(config), only_stages=only_stages,
    )
    _print_enrichment_summary(report)
    _handle_rechunk_offer(config, report, rechunk=rechunk, skip_rechunk=skip_rechunk)
    return 0


def _print_enrichment_summary(report) -> None:
    """Print a human-readable enrichment summary."""
    print(f"  Tier 1 enriched: {report.tier1_updated:,} (simhash, defaults, tags)")
    print(f"  Tier 2 enriched: {report.tier2_updated:,} (heading_path, positions, graph)")
    print(f"  Tier 2b enriched: {report.tier2b_updated:,} (cluster rebuild)")
    if report.tier2_skipped:
        print(f"  Tier 2 skipped: {len(report.tier2_skipped):,} (source files not found)")
    # Separate drifted files from other tier3 items
    drifted_count = len(report.drifted_source_paths)
    other_tier3 = [t for t in report.tier3_needed if not t.startswith("boundary drifted:")]
    if drifted_count:
        print(f"  Boundary-drifted files: {drifted_count:,} (old chunking strategy, can be rechunked)")
    if other_tier3:
        print(f"  Missing summaries: {len(other_tier3):,} (need re-ingest)")
        print(
            "\nWARNING: Some files need re-ingest (missing summaries).\n"
            "Run `lsm ingest --force-reingest-changed-config` to re-chunk, re-embed, and regenerate summaries."
        )
    if report.errors:
        for err in report.errors:
            print(f"  Error: {err}")


def _handle_rechunk_offer(
    config: LSMConfig,
    report,
    *,
    rechunk: bool = False,
    skip_rechunk: bool = False,
) -> None:
    """Offer to rechunk boundary-drifted files after enrichment.

    Behaviour:
    - ``--skip-rechunk``: print skip message, return.
    - ``--rechunk``: proceed automatically.
    - Neither: interactive ``[y/N]`` prompt.
    """
    drifted = report.drifted_source_paths
    if not drifted:
        return

    # Filter to paths that still exist on disk
    valid_paths = {p for p in drifted if Path(p).exists()}
    if not valid_paths:
        print(
            f"\n{len(drifted)} boundary-drifted file(s) detected, "
            "but none exist on disk. Skipping rechunk."
        )
        return

    print(
        f"\n{len(valid_paths)} boundary-drifted file(s) can be rechunked with the "
        "current structure-aware chunking strategy."
    )

    if skip_rechunk:
        print("Skipping rechunk (--skip-rechunk).")
        return

    if not rechunk:
        try:
            answer = input("Rechunk these files now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping rechunk.")
            return
        if answer not in ("y", "yes"):
            print("Skipping rechunk.")
            return

    print(f"Rechunking {len(valid_paths)} file(s)...")
    try:
        result = api_run_ingest(config, force_source_paths=valid_paths)
        print(
            f"Rechunk complete: {result.completed_files} file(s) processed, "
            f"{result.chunks_added} chunk(s) added."
        )
    except Exception as exc:
        print(f"Rechunk error: {exc}")


def run_cache_clear_cli(
    config: LSMConfig,
    *,
    clear_reranker: bool = False,
) -> int:
    """Clear reranker cache."""
    if not clear_reranker:
        clear_reranker = True

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", getattr(provider, "_conn", None))
    tn = _table_names(config)
    if conn is None:
        print("Error: Reranker cache clear requires a backend with direct SQL access.")
        return 1
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tn.reranker_cache} (
                cache_key TEXT PRIMARY KEY,
                score REAL,
                created_at TEXT
            )
            """
        )
        row = conn.execute(
            f"SELECT COUNT(*) FROM {tn.reranker_cache}"
        ).fetchone()
        removed = int(row[0]) if row else 0
        conn.execute(f"DELETE FROM {tn.reranker_cache}")
        conn.commit()
    except Exception as exc:
        print(f"Error clearing reranker cache: {exc}")
        return 1

    print(f"Cleared reranker cache entries: {removed}")
    return 0


def run_cluster(args, config: LSMConfig) -> int:
    """Run cluster management commands."""
    command = getattr(args, "cluster_command", None)
    if command == "build":
        return run_cluster_build_cli(
            config,
            algorithm=getattr(args, "algorithm", None),
            k=getattr(args, "k", None),
        )
    if command == "visualize":
        return run_cluster_visualize_cli(
            config,
            output=getattr(args, "output", "clusters.html"),
        )

    print("Missing cluster subcommand. Use `lsm cluster --help` for options.")
    return 2


def run_cluster_build_cli(
    config: LSMConfig,
    algorithm: Optional[str] = None,
    k: Optional[int] = None,
) -> int:
    """Build cluster assignments for all current embeddings."""
    from lsm.db.clustering import build_clusters
    from lsm.db.job_status import record_job_status

    algorithm = algorithm or config.query.cluster_algorithm
    k = k or config.query.cluster_k
    tn = _table_names(config)

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    if conn is None:
        print("Error: Clustering requires SQLite backend with direct connection access.")
        return 1

    print(f"Building clusters: algorithm={algorithm}, k={k}")
    try:
        result = build_clusters(conn, algorithm=algorithm, k=k, table_names=tn)
    except ImportError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error during clustering: {exc}")
        return 1

    print(
        f"Clustering complete: {result['n_clusters']} clusters, "
        f"{result['n_chunks']} chunks, algorithm={result['algorithm']}"
    )
    try:
        record_job_status(
            conn,
            "cluster_build",
            corpus_size=int(result.get("n_chunks", 0)),
        )
    except Exception:
        logger.debug("Failed to record cluster_build job status", exc_info=True)
    return 0


def run_cluster_visualize_cli(
    config: LSMConfig,
    output: str = "clusters.html",
) -> int:
    """Export a UMAP HTML plot of cluster distributions."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required for cluster visualization.")
        return 1
    try:
        import umap  # noqa: F401
    except ImportError:
        print(
            "Error: umap-learn is required for cluster visualization.\n"
            "Install with: pip install 'lsm[clustering]'"
        )
        return 1

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    tn = _table_names(config)
    if conn is None:
        print("Error: Cluster visualization requires SQLite backend with direct connection access.")
        return 1

    print("Loading embeddings and cluster assignments...")
    try:
        rows = conn.execute(
            f"SELECT c.chunk_id, c.cluster_id, v.embedding "
            f"FROM {tn.chunks} c "
            f"JOIN {tn.vec_chunks} v ON c.chunk_id = v.chunk_id "
            f"WHERE c.is_current = 1 AND c.cluster_id IS NOT NULL"
        ).fetchall()
    except Exception as exc:
        print(f"Error loading data: {exc}")
        return 1

    if not rows:
        print("No clustered chunks found. Run `lsm cluster build` first.")
        return 1

    from struct import unpack

    chunk_ids = [r[0] for r in rows]
    cluster_ids = [int(r[1]) for r in rows]
    embeddings = []
    for r in rows:
        raw = r[2]
        dim = len(raw) // 4
        embeddings.append(list(unpack(f"{dim}f", raw)))
    embeddings_np = np.array(embeddings, dtype=np.float32)

    print(f"Running UMAP on {len(chunk_ids)} chunks...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings_np)

    # Generate HTML with inline scatter plot
    unique_clusters = sorted(set(cluster_ids))
    colors = [
        f"hsl({int(360 * i / max(len(unique_clusters), 1))}, 70%, 50%)"
        for i in range(len(unique_clusters))
    ]
    cluster_color = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

    points_js = ",".join(
        f'{{x:{coords[i][0]:.4f},y:{coords[i][1]:.4f},c:{cluster_ids[i]}}}'
        for i in range(len(chunk_ids))
    )

    html = f"""<!DOCTYPE html>
<html><head><title>LSM Cluster Visualization</title>
<style>body{{font-family:sans-serif;margin:20px}}canvas{{border:1px solid #ccc}}</style>
</head><body>
<h2>LSM Cluster Visualization ({len(chunk_ids)} chunks, {len(unique_clusters)} clusters)</h2>
<canvas id="plot" width="800" height="600"></canvas>
<script>
var data=[{points_js}];
var colors={{{",".join(f"{cid}:'{c}'" for cid, c in cluster_color.items())}}};
var canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
var xs=data.map(d=>d.x),ys=data.map(d=>d.y);
var xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
var pad=40,w=canvas.width-2*pad,h=canvas.height-2*pad;
data.forEach(function(d){{
  var px=pad+(d.x-xmin)/(xmax-xmin)*w,py=pad+(d.y-ymin)/(ymax-ymin)*h;
  ctx.beginPath();ctx.arc(px,py,3,0,2*Math.PI);
  ctx.fillStyle=colors[d.c]||'#999';ctx.fill();
}});
</script></body></html>"""

    Path(output).write_text(html, encoding="utf-8")
    print(f"Cluster visualization saved to {output}")
    return 0


def run_finetune(args, config: LSMConfig) -> int:
    """Run fine-tuning commands."""
    command = getattr(args, "finetune_command", None)
    if command == "train":
        return run_finetune_train_cli(
            config,
            base_model=getattr(args, "base_model", "sentence-transformers/all-MiniLM-L6-v2"),
            epochs=getattr(args, "epochs", 3),
            output=getattr(args, "output", "./models/finetuned"),
            max_pairs=getattr(args, "max_pairs", None),
        )
    if command == "list":
        return run_finetune_list_cli(config)
    if command == "activate":
        return run_finetune_activate_cli(config, model_id=args.model_id)

    print("Missing finetune subcommand. Use `lsm finetune --help` for options.")
    return 2


def run_finetune_train_cli(
    config: LSMConfig,
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    output: str = "./models/finetuned",
    max_pairs: Optional[int] = None,
) -> int:
    """Extract training pairs from corpus and fine-tune embedding model."""
    from lsm.finetune.embedding import extract_training_pairs, finetune_embedding_model
    from lsm.finetune.registry import register_model, set_active_model
    from lsm.db.job_status import record_job_status

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    tn = _table_names(config)
    if conn is None:
        print("Error: Fine-tuning requires SQLite backend with direct connection access.")
        return 1

    print("Extracting training pairs from corpus...")
    pairs = extract_training_pairs(conn, max_pairs=max_pairs, table_names=tn)
    if not pairs:
        print("Error: No training pairs found. Ingest data first.")
        return 1
    print(f"Found {len(pairs)} training pairs.")

    print(f"Fine-tuning {base_model} for {epochs} epochs...")
    try:
        result = finetune_embedding_model(
            pairs=pairs,
            base_model=base_model,
            output_path=output,
            epochs=epochs,
        )
    except ImportError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        print(f"Error during fine-tuning: {exc}")
        return 1

    # Register in model registry
    entry = register_model(
        conn=conn,
        model_id=result["model_id"],
        base_model=result["base_model"],
        path=result["output_path"],
        dimension=result["dimension"],
        table_names=tn,
    )
    set_active_model(conn, entry.model_id, table_names=tn)

    print(f"Fine-tuning complete:")
    print(f"  Model ID: {entry.model_id}")
    print(f"  Path: {entry.path}")
    print(f"  Dimension: {entry.dimension}")
    print(f"  Training pairs: {result['num_pairs']}")
    print(f"  Status: active")

    try:
        row = conn.execute(
            f"SELECT COUNT(*) FROM {tn.chunks} WHERE is_current = 1"
        ).fetchone()
        corpus_size = int(row[0]) if row else None
        record_job_status(conn, "finetune_embedding", corpus_size=corpus_size)
    except Exception:
        logger.debug("Failed to record finetune_embedding job status", exc_info=True)

    return 0


def run_finetune_list_cli(config: LSMConfig) -> int:
    """List registered fine-tuned models."""
    from lsm.finetune.registry import list_models

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    if conn is None:
        print("Error: Model registry requires SQLite backend.")
        return 1

    tn = _table_names(config)
    models = list_models(conn, table_names=tn)
    if not models:
        print("No fine-tuned models registered.")
        return 0

    print(f"{'Model ID':<30} {'Base Model':<35} {'Dim':>5} {'Active':>7} {'Created'}")
    print("-" * 100)
    for m in models:
        active = "  *" if m.is_active else ""
        print(f"{m.model_id:<30} {m.base_model:<35} {m.dimension:>5} {active:>7} {m.created_at}")
    return 0


def run_finetune_activate_cli(config: LSMConfig, model_id: str) -> int:
    """Set a fine-tuned model as active."""
    from lsm.finetune.registry import get_active_model, list_models, set_active_model

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    if conn is None:
        print("Error: Model registry requires SQLite backend.")
        return 1

    # Check model exists
    tn = _table_names(config)
    models = list_models(conn, table_names=tn)
    ids = [m.model_id for m in models]
    if model_id not in ids:
        print(f"Error: Model '{model_id}' not found. Available: {', '.join(ids) or 'none'}")
        return 1

    set_active_model(conn, model_id, table_names=tn)
    print(f"Activated model: {model_id}")
    return 0


def run_graph(args, config: LSMConfig) -> int:
    """Run graph management commands."""
    command = getattr(args, "graph_command", None)
    if command == "build-links":
        return run_graph_build_links_cli(
            config,
            threshold=getattr(args, "threshold", 0.8),
            batch_size=getattr(args, "batch_size", 500),
        )

    print("Missing graph subcommand. Use `lsm graph --help` for options.")
    return 2


def run_graph_build_links_cli(
    config: LSMConfig,
    threshold: float = 0.8,
    batch_size: int = 500,
) -> int:
    """Build thematic links between chunks using embedding cosine similarity."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required for graph link building.")
        return 1

    provider = create_vectordb_provider(config.db)
    conn = getattr(provider, "connection", None)
    tn = _table_names(config)
    if conn is None:
        print("Error: Graph link building requires SQLite backend with direct connection access.")
        return 1

    print(f"Building thematic links: threshold={threshold}, batch_size={batch_size}")

    # Load all current chunk embeddings
    try:
        from struct import unpack

        rows = conn.execute(
            f"SELECT c.chunk_id, v.embedding "
            f"FROM {tn.chunks} c "
            f"JOIN {tn.vec_chunks} v ON c.chunk_id = v.chunk_id "
            f"WHERE c.is_current = 1"
        ).fetchall()
    except Exception as exc:
        print(f"Error loading embeddings: {exc}")
        return 1

    if not rows:
        print("No chunks found. Run `lsm ingest build` first.")
        return 1

    chunk_ids = [r[0] for r in rows]
    embeddings = []
    for r in rows:
        raw = r[1]
        dim = len(raw) // 4
        embeddings.append(list(unpack(f"{dim}f", raw)))
    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    print(f"Comparing {len(chunk_ids)} chunks in batches of {batch_size}...")
    edges_created = 0

    for i in range(0, len(chunk_ids), batch_size):
        batch_end = min(i + batch_size, len(chunk_ids))
        batch = emb_matrix[i:batch_end]

        # Compare this batch against all chunks after it to avoid duplicates
        for j in range(i, len(chunk_ids), batch_size):
            compare_end = min(j + batch_size, len(chunk_ids))
            compare = emb_matrix[j:compare_end]

            sim = batch @ compare.T

            for bi in range(batch.shape[0]):
                for ci in range(compare.shape[0]):
                    global_i = i + bi
                    global_j = j + ci
                    if global_i >= global_j:
                        continue  # skip self and already-seen pairs
                    if sim[bi, ci] >= threshold:
                        src_id = chunk_ids[global_i]
                        dst_id = chunk_ids[global_j]
                        conn.execute(
                            f"INSERT OR IGNORE INTO {tn.graph_edges} "
                            "(src_id, dst_id, edge_type, weight) "
                            "VALUES (?, ?, 'thematic', ?)",
                            (src_id, dst_id, float(sim[bi, ci])),
                        )
                        edges_created += 1

    conn.commit()
    print(f"Link building complete: {edges_created} thematic edges created.")
    try:
        from lsm.db.job_status import record_job_status

        record_job_status(
            conn,
            "graph_build_links",
            corpus_size=len(chunk_ids),
        )
    except Exception:
        logger.debug("Failed to record graph_build_links job status", exc_info=True)
    return 0
