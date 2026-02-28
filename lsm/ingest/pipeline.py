from __future__ import annotations

import os
import time
import threading
import queue
import json
import fnmatch
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from lsm.ingest.chunking import chunk_text
from lsm.ingest.fs import iter_files, collect_folder_tags
from lsm.ingest.manifest import (
    get_next_version,
    load_manifest,
    save_manifest,
    upsert_manifest_entries,
)
from lsm.ingest.models import PageSegment, ParseResult, WriteJob
from lsm.ingest.parsers import parse_file
from lsm.ingest.structure_chunking import structure_chunk_text, structured_chunks_to_positions
from lsm.ingest.utils import (
    canonical_path,
    file_sha256,
    now_iso,
    normalize_whitespace,
    format_time,
    make_chunk_id,
)
from lsm.db.completion import detect_completion_mode, get_stale_files
from lsm.db.schema_version import (
    SchemaVersionMismatchError,
    check_schema_compatibility,
    record_schema_version,
)
from lsm.vectordb import BaseVectorDBProvider, create_vectordb_provider
from lsm.config.models import LLMConfig, RootConfig, VectorDBConfig

# -----------------------------
# Main ingest pipeline
# -----------------------------
def parse_and_chunk_job(
    fp: Path,
    source_path: str,
    mtime_ns: int,
    size: int,
    fhash: str,
    had_prev: bool,
    enable_ocr: bool = False,
    skip_errors: bool = True,
    stop_event: Optional[threading.Event] = None,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    chunking_strategy: str = "structure",
    enable_language_detection: bool = False,
    enable_translation: bool = False,
    translation_target: str = "en",
    translation_llm_config: Optional[LLMConfig] = None,
    root_tags: Optional[List[str]] = None,
    content_type: Optional[str] = None,
    folder_tags: Optional[List[str]] = None,
) -> ParseResult:
    """
    Parse a file, extract metadata, chunk the text, and track positions.

    Args:
        fp: Path to file
        source_path: Canonical source path
        mtime_ns: Modified time in nanoseconds
        size: File size in bytes
        fhash: SHA-256 hash of file
        had_prev: Whether file was previously ingested
        enable_ocr: Enable OCR for image-based PDFs
        skip_errors: If True, continue on per-file failures
        stop_event: Threading event for graceful shutdown
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap in characters
        chunking_strategy: 'structure' for heading/paragraph/sentence-aware
            chunking, 'fixed' for simple sliding-window chunking.
        enable_language_detection: Detect document language and store in metadata.
        enable_translation: Translate non-target-language chunks via LLM.
        translation_target: Target language code for translation (ISO 639-1).
        translation_llm_config: LLM config for translation service.
        root_tags: Tags from the root config this file belongs to.
        content_type: Content type label from the root config.
        folder_tags: Tags collected from .lsm_tags.json files in parent dirs.

    Returns:
        ParseResult with text chunks, metadata, and positions
    """
    try:
        if stop_event and stop_event.is_set():
            return ParseResult(
                source_path=source_path,
                fp=fp,
                mtime_ns=mtime_ns,
                size=size,
                file_hash=fhash,
                chunks=[],
                ext=fp.suffix.lower(),
                had_prev=had_prev,
                ok=False,
                err="stopped",
            )
        # Parse file and extract metadata + page segments
        raw_text, doc_metadata, page_segments = parse_file(
            fp, enable_ocr=enable_ocr, skip_errors=skip_errors,
        )

        parse_errors: List[Dict[str, Any]] = []
        if doc_metadata and "_parse_errors" in doc_metadata:
            parse_errors = doc_metadata.pop("_parse_errors") or []

        # Normalize whitespace
        raw_text = normalize_whitespace(raw_text)

        # Detect language if enabled
        if enable_language_detection:
            from lsm.ingest.language import detect_language_for_document

            detected_lang = detect_language_for_document(raw_text)
            if detected_lang:
                if doc_metadata is None:
                    doc_metadata = {}
                doc_metadata["language"] = detected_lang

        # Inject root/folder tags into document metadata
        if root_tags or content_type or folder_tags:
            if doc_metadata is None:
                doc_metadata = {}
            if root_tags:
                doc_metadata["root_tags"] = json.dumps(root_tags)
            if content_type:
                doc_metadata["content_type"] = content_type
            if folder_tags:
                doc_metadata["folder_tags"] = json.dumps(folder_tags)

        if not raw_text:
            return ParseResult(
                source_path=source_path,
                fp=fp,
                mtime_ns=mtime_ns,
                size=size,
                file_hash=fhash,
                chunks=[],
                ext=fp.suffix.lower(),
                had_prev=had_prev,
                ok=False,
                err="empty_text",
                metadata=doc_metadata,
                parse_errors=parse_errors,
            )

        # Chunk text using the configured strategy
        if chunking_strategy == "structure":
            # Convert overlap from absolute chars to a ratio for structure chunking
            overlap_ratio = chunk_overlap / chunk_size if chunk_size > 0 else 0.0
            structured = structure_chunk_text(
                raw_text,
                chunk_size=chunk_size,
                overlap=overlap_ratio,
                page_segments=page_segments,
                track_positions=True,
            )
            chunks, positions = structured_chunks_to_positions(structured)
        else:
            # Fixed (legacy) chunking
            chunks, positions = chunk_text(
                raw_text,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                track_positions=True,
            )

        if not chunks:
            return ParseResult(
                source_path=source_path,
                fp=fp,
                mtime_ns=mtime_ns,
                size=size,
                file_hash=fhash,
                chunks=[],
                ext=fp.suffix.lower(),
                had_prev=had_prev,
                ok=False,
                err="no_chunks",
                metadata=doc_metadata,
                parse_errors=parse_errors,
            )

        # Translate chunks if enabled and language differs from target
        if enable_translation and translation_llm_config is not None:
            detected_lang = (doc_metadata or {}).get("language")
            if detected_lang and detected_lang != translation_target:
                from lsm.ingest.translation import translate_chunk

                translated_chunks = []
                for chunk in chunks:
                    translated = translate_chunk(
                        text=chunk,
                        target_lang=translation_target,
                        llm_config=translation_llm_config,
                        source_lang=detected_lang,
                    )
                    translated_chunks.append(translated)
                if doc_metadata is None:
                    doc_metadata = {}
                doc_metadata["translated_from"] = detected_lang
                chunks = translated_chunks

        return ParseResult(
            source_path=source_path,
            fp=fp,
            mtime_ns=mtime_ns,
            size=size,
            file_hash=fhash,
            chunks=chunks,
            ext=fp.suffix.lower(),
            had_prev=had_prev,
            ok=True,
            metadata=doc_metadata,
            chunk_positions=positions,
            parse_errors=parse_errors,
            page_segments=page_segments,
        )

    except Exception as e:
        if not skip_errors:
            raise
        return ParseResult(
            source_path=source_path,
            fp=fp,
            mtime_ns=mtime_ns,
            size=size,
            file_hash=fhash,
            chunks=[],
            ext=fp.suffix.lower(),
            had_prev=had_prev,
            ok=False,
            err=str(e),
        )


def _resolve_runtime_artifact_dir(vectordb_config: VectorDBConfig) -> Path:
    vdb_path = Path(vectordb_config.path)
    if str(vdb_path).lower().endswith(".db"):
        return vdb_path.parent
    return vdb_path


def ingest(
    roots: List[RootConfig],
    chroma_flush_interval: Optional[int],
    embed_model_name: str,
    device: str,
    batch_size: int,
    manifest_path: Optional[Path],
    exts: set[str],
    exclude_dirs: set[str],
    vectordb_config: VectorDBConfig,
    dry_run: bool = False,
    enable_ocr: bool = False,
    skip_errors: bool = True,
    stop_event: Optional[threading.Event] = None,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
    chunking_strategy: str = "structure",
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    enable_language_detection: bool = False,
    enable_translation: bool = False,
    translation_target: str = "en",
    translation_llm_config: Optional[LLMConfig] = None,
    embedding_dimension: Optional[int] = None,
    max_files: Optional[int] = None,
    max_seconds: Optional[int] = None,
    enable_versioning: bool = True,
    force_reingest: bool = False,
    force_reingest_changed_config: bool = False,
    force_file_pattern: Optional[str] = None,
    provider: Optional[BaseVectorDBProvider] = None,
) -> Dict[str, Any]:
    """
    Run ingest pipeline.

    Returns:
        Summary dictionary with ingest metrics.
    """
    _ = enable_versioning  # Versioning is always on; retained for compatibility.

    def emit(event: str, current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(event, current, total, message)

    stop_signal = stop_event or threading.Event()
    provider = provider or create_vectordb_provider(vectordb_config)
    manifest_connection = getattr(provider, "connection", None)
    if not isinstance(manifest_connection, sqlite3.Connection):
        manifest_connection = None

    emit("init", 0, 0, f"Model device: {device}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model_name, device=device)

    # Validate embedding dimension
    actual_dim = model.get_sentence_embedding_dimension()
    if embedding_dimension is not None:
        if actual_dim != embedding_dimension:
            raise ValueError(
                f"Embedding model '{embed_model_name}' produces "
                f"{actual_dim}-dimensional vectors, but "
                f"embedding_dimension is configured as {embedding_dimension}. "
                f"Update global.embedding_dimension or change the model."
            )
    else:
        emit("init", 0, 0, f"Auto-detected embedding dimension: {actual_dim}")

    schema_config = {
        "embedding_model": embed_model_name,
        "embedding_dim": actual_dim,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    schema_version_id: Optional[int] = None
    completion_mode: Optional[str] = None
    stale_file_paths: Optional[set[str]] = None
    if manifest_connection is not None:
        compatible, diff = check_schema_compatibility(
            manifest_connection,
            schema_config,
            raise_on_mismatch=False,
        )
        completion_runtime_config = {
            "roots": roots,
            "exts": exts,
            "exclude_dirs": exclude_dirs,
            **schema_config,
        }
        if force_reingest_changed_config:
            completion_mode = detect_completion_mode(
                manifest_connection,
                completion_runtime_config,
            )
            if completion_mode:
                stale_file_paths = set(
                    get_stale_files(
                        manifest_connection,
                        completion_runtime_config,
                        completion_mode,
                    )
                )
            elif not compatible:
                stale_file_paths = set()
        elif not compatible:
            raise SchemaVersionMismatchError(diff)
        schema_version_id = record_schema_version(manifest_connection, schema_config)

    if force_reingest:
        manifest = {}
        if manifest_connection is not None:
            load_manifest(connection=manifest_connection)
            manifest_connection.execute("DELETE FROM lsm_manifest")
            manifest_connection.commit()
        elif manifest_path is not None and manifest_path.exists():
            manifest_path.unlink()
    else:
        manifest = load_manifest(manifest_path, connection=manifest_connection)

    if completion_mode is not None:
        emit(
            "init",
            0,
            0,
            "Completion mode detected: "
            f"{completion_mode} (stale files: {len(stale_file_paths or set())})",
        )
    if force_file_pattern:
        emit("init", 0, 0, f"Selective ingest pattern enabled: {force_file_pattern}")

    runtime_artifact_dir = _resolve_runtime_artifact_dir(vectordb_config)
    runtime_artifact_dir.mkdir(parents=True, exist_ok=True)

    file_tuples = list(iter_files(roots, exts, exclude_dirs))
    total_files = len(file_tuples)
    emit("discovery", 0, total_files, f"Discovered {total_files:,} files to consider")

    # ---- Queues (bounded for backpressure) ----
    parse_out_q: "queue.Queue[Optional[ParseResult]]" = queue.Queue(maxsize=128)
    write_q: "queue.Queue[Optional[WriteJob]]" = queue.Queue(maxsize=64)

    # ---- Metrics (thread-safe) ----
    lock = threading.Lock()
    start_time = time.time()
    last_report = start_time

    scanned = 0
    processed = 0
    completed = 0
    skipped = 0
    embedded_files = 0
    added_chunks = 0
    embed_seconds = 0.0
    files_submitted = 0
    limit_reached = False
    written_chunks = 0
    writer_error: Optional[Exception] = None
    error_report_path = runtime_artifact_dir / "ingest_error_report.json"
    error_records = {
        "failed_documents": [],
        "page_errors": [],
    }

    REPORT_EVERY_SECONDS = 1.0
    flush_threshold = max(64, int(chroma_flush_interval or (batch_size * 8)))
    try:
        from lsm.vectordb.sqlite_vec import SQLiteVecProvider  # local import to avoid hard dependency in tests

        is_sqlite_provider = isinstance(provider, SQLiteVecProvider)
    except Exception:
        is_sqlite_provider = False
    transactional_manifest_writes = (
        not dry_run
        and manifest_connection is not None
        and is_sqlite_provider
    )

    # ---- Writer thread (sole Chroma owner) ----
    def writer_thread():
        nonlocal written_chunks, writer_error

        try:
            to_add_ids: List[str] = []
            to_add_docs: List[str] = []
            to_add_metas: List[Dict] = []
            to_add_embs: List[List[float]] = []
            seen_ids: set[str] = set()
            # Manifest updates are committed only after a successful vector DB flush.
            pending_manifest_updates: Dict[str, Dict] = {}

            def flush():
                """Flush staged vectors and, only on success, commit pending manifest updates."""
                nonlocal to_add_ids, to_add_docs, to_add_metas, to_add_embs, seen_ids, pending_manifest_updates, written_chunks

                if not to_add_ids:
                    return

                if not dry_run:
                    if transactional_manifest_writes and manifest_connection is not None:
                        manifest_connection.execute("BEGIN")
                        try:
                            provider.add_chunks(to_add_ids, to_add_docs, to_add_metas, to_add_embs)
                            upsert_manifest_entries(
                                manifest_connection,
                                pending_manifest_updates,
                                commit=False,
                            )
                            manifest_connection.commit()
                        except Exception:
                            manifest_connection.rollback()
                            raise
                    else:
                        # If this raises, we do NOT update the manifest.
                        provider.add_chunks(to_add_ids, to_add_docs, to_add_metas, to_add_embs)

                # Commit manifest updates only after successful write (or dry_run)
                manifest.update(pending_manifest_updates)
                pending_manifest_updates.clear()

                with lock:
                    written_chunks += len(to_add_ids)

                to_add_ids, to_add_docs, to_add_metas, to_add_embs = [], [], [], []
                seen_ids.clear()

            while True:
                if stop_signal.is_set() and write_q.empty():
                    break
                try:
                    job = write_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if job is None:
                    break

                # Handle prior chunks
                if job.had_prev:
                    # Versioning is always on in v0.8.0.
                    try:
                        old = provider.get(
                            filters={"source_path": job.source_path}, include=["metadatas"],
                        )
                        old_ids = old.ids
                        if old_ids:
                            updated = [{"is_current": False} for _ in old_ids]
                            provider.update_metadatas(old_ids, updated)
                    except Exception:
                        pass

                ingested_at = now_iso()

                # Stage chunks
                for idx, (chunk, emb) in enumerate(zip(job.chunks, job.embeddings)):
                    chunk_id = make_chunk_id(job.source_path, job.file_hash, idx)
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)

                    # Base metadata
                    meta = {
                        "source_path": job.source_path,
                        "source_name": job.fp.name,
                        "ext": job.ext,
                        "mtime_ns": job.mtime_ns,
                        "file_hash": job.file_hash,
                        "chunk_index": idx,
                        "ingested_at": ingested_at,
                    }

                    # Merge document-level metadata
                    if job.metadata:
                        for key, value in job.metadata.items():
                            # Avoid overwriting base fields
                            if key not in meta and value is not None:
                                meta[key] = value

                    # Add position information
                    if job.chunk_positions and idx < len(job.chunk_positions):
                        pos = job.chunk_positions[idx]
                        meta["start_char"] = pos.get("start_char")
                        meta["end_char"] = pos.get("end_char")
                        meta["chunk_length"] = pos.get("length")
                        # Structure chunking metadata
                        if pos.get("heading") is not None:
                            meta["heading"] = pos["heading"]
                        if pos.get("paragraph_index") is not None:
                            meta["paragraph_index"] = pos["paragraph_index"]
                        # Page number tracking
                        if pos.get("page_start") is not None and pos.get("page_end") is not None:
                            if pos["page_start"] == pos["page_end"]:
                                meta["page_number"] = str(pos["page_start"])
                            else:
                                meta["page_number"] = f"{pos['page_start']}-{pos['page_end']}"

                    # Versioning metadata is always set in v0.8.0.
                    meta["is_current"] = True
                    meta["version"] = job.version

                    to_add_ids.append(chunk_id)
                    to_add_docs.append(chunk)
                    to_add_metas.append(meta)
                    to_add_embs.append(emb)

                # Stage manifest update for this file, but do NOT commit yet
                manifest_entry = {
                    "mtime_ns": job.mtime_ns,
                    "size": job.size,
                    "file_hash": job.file_hash,
                    "version": job.version,
                    "embedding_model": embed_model_name,
                    "schema_version_id": schema_version_id,
                    "updated_at": now_iso(),
                }
                pending_manifest_updates[job.source_path] = manifest_entry

                # Flush when we hit threshold
                if len(to_add_ids) >= flush_threshold:
                    flush()

            # Final flush
            flush()
        except Exception as exc:
            writer_error = exc
            stop_signal.set()

    wt = threading.Thread(target=writer_thread, daemon=True)
    wt.start()

    # ---- Embed worker (single GPU consumer; batches across files) ----
    def embed_worker():
        nonlocal embed_seconds, embedded_files, added_chunks

        pending: List[ParseResult] = []
        pending_chunk_count = 0
        MAX_CHUNKS_PER_GPU_BATCH = max(batch_size * 16, 1024)  # tune: larger => better GPU utilization

        def flush_pending():
            nonlocal embed_seconds, embedded_files, added_chunks, pending_chunk_count
            if not pending:
                return

            # Flatten chunks
            all_chunks: List[str] = []
            offsets: List[Tuple[int, int]] = []  # (start, end) per file
            cursor = 0
            for pr in pending:
                start = cursor
                all_chunks.extend(pr.chunks)
                cursor += len(pr.chunks)
                offsets.append((start, cursor))

            # GPU embed
            t0 = time.time()
            embs = model.encode(
                all_chunks,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            dt = time.time() - t0

            # Convert to list-of-lists efficiently
            # sentence-transformers returns numpy ndarray by default
            embs_list: List[List[float]] = embs.tolist()

            with lock:
                embed_seconds += dt

            # Emit per-file write jobs
            for pr, (a, b) in zip(pending, offsets):
                wj = WriteJob(
                    source_path=pr.source_path,
                    fp=pr.fp,
                    mtime_ns=pr.mtime_ns,
                    size=pr.size,
                    file_hash=pr.file_hash,
                    ext=pr.ext,
                    chunks=pr.chunks,
                    embeddings=embs_list[a:b],
                    had_prev=pr.had_prev,
                    metadata=pr.metadata,
                    chunk_positions=pr.chunk_positions,
                    version=pr.version,
                )
                write_q.put(wj)

                with lock:
                    embedded_files += 1
                    added_chunks += len(pr.chunks)

            pending.clear()
            pending_chunk_count = 0 

        while True:
            if stop_signal.is_set():
                break
            try:
                pr = parse_out_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if pr is None:
                break

            if not pr.ok:
                with lock:
                    pass
                continue

            pending.append(pr)
            pending_chunk_count += len(pr.chunks)

            if pending_chunk_count >= MAX_CHUNKS_PER_GPU_BATCH:
                flush_pending()
                pending_chunk_count = 0

        # Flush remaining
        if not stop_signal.is_set():
            flush_pending()

        # Signal writer shutdown
        write_q.put(None)

    et = threading.Thread(target=embed_worker, daemon=True)
    et.start()

    # ---- Parse thread pool ----
    # Good starting point: 4–12 threads. PDFs are expensive; too many threads can thrash disk.
    parse_workers = min(12, max(4, (os.cpu_count() or 8) // 2))

    def maybe_report():
        nonlocal last_report
        now = time.time()
        if (now - last_report) < REPORT_EVERY_SECONDS:
            return
        elapsed = now - start_time

        with lock:
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = total_files - completed
            eta = (remaining / rate) if rate > 0 else float("inf")
            chunk_rate = (added_chunks / embed_seconds) if embed_seconds > 0 else 0.0
            wc = written_chunks

        emit(
            "progress",
            completed,
            total_files,
            (
                f"queued={processed:,} skipped={skipped:,} embedded_files={embedded_files:,} "
                f"chunks_added={added_chunks:,} chunks_written={wc:,} "
                f"file_rate={rate:0.2f}/s embed_rate={chunk_rate:0.1f}/s "
                f"elapsed={format_time(elapsed)} eta={format_time(eta)}"
            ),
        )
        last_report = now

    force_pattern = str(force_file_pattern or "").strip()
    if not force_pattern:
        force_pattern = ""

    try:
        with ThreadPoolExecutor(max_workers=parse_workers) as pool:
            futures = []

            for fp, root_cfg in file_tuples:
                if stop_signal.is_set():
                    break
                # Check time limit before processing next file
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    emit("limit", files_submitted, total_files,
                         f"Reached max_seconds limit ({max_seconds}s)")
                    limit_reached = True
                    break
                scanned += 1
                source_path = canonical_path(fp)
                key = source_path

                if force_pattern:
                    match_pattern = (
                        fnmatch.fnmatch(source_path.lower(), force_pattern.lower())
                        or fnmatch.fnmatch(fp.name.lower(), force_pattern.lower())
                    )
                    if not match_pattern:
                        skipped += 1
                        processed += 1
                        completed += 1
                        maybe_report()
                        continue

                if stale_file_paths is not None and key not in stale_file_paths:
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                # Stat
                try:
                    st = fp.stat()
                    mtime_ns = st.st_mtime_ns
                    size = st.st_size
                except Exception:
                    skipped += 1
                    processed += 1
                    completed += 1
                    continue

                prev = manifest.get(key)
                force_this_file = bool(force_pattern) or (
                    stale_file_paths is not None and key in stale_file_paths
                )

                # Fast skip on (mtime,size) + hash present
                if (
                    not force_this_file
                    and prev
                    and prev.get("mtime_ns") == mtime_ns
                    and prev.get("size") == size
                    and prev.get("file_hash")
                ):
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                # Hash only when needed
                try:
                    fhash = file_sha256(fp)
                except Exception:
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                if (
                    force_this_file
                    and completion_mode == "metadata_enrichment"
                    and prev
                    and prev.get("file_hash") == fhash
                ):
                    existing = provider.get(
                        filters={"source_path": key},
                        include=["metadatas"],
                    )
                    existing_ids = list(existing.ids or [])
                    existing_metas = list(existing.metadatas or [])
                    if existing_ids:
                        while len(existing_metas) < len(existing_ids):
                            existing_metas.append({})
                        folder_tags = collect_folder_tags(fp, root_cfg.path)
                        updated_metas: List[Dict[str, Any]] = []
                        for metadata in existing_metas[: len(existing_ids)]:
                            merged = dict(metadata or {})
                            if root_cfg.tags:
                                merged["root_tags"] = json.dumps(root_cfg.tags)
                            if root_cfg.content_type:
                                merged["content_type"] = root_cfg.content_type
                            if folder_tags:
                                merged["folder_tags"] = json.dumps(folder_tags)
                            updated_metas.append(merged)
                        provider.update_metadatas(existing_ids, updated_metas)

                    manifest[key] = {
                        "mtime_ns": mtime_ns,
                        "size": size,
                        "file_hash": fhash,
                        "version": int(prev.get("version", 1)),
                        "embedding_model": embed_model_name,
                        "schema_version_id": schema_version_id,
                        "updated_at": now_iso(),
                    }
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                if not force_this_file and prev and prev.get("file_hash") == fhash:
                    # content unchanged, update manifest
                    manifest[key] = {
                        "mtime_ns": mtime_ns,
                        "size": size,
                        "file_hash": fhash,
                        "version": int(prev.get("version", 1)) if prev else 1,
                        "embedding_model": embed_model_name,
                        "schema_version_id": schema_version_id,
                        "updated_at": now_iso(),
                    }
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                had_prev = prev is not None

                # Versioning is always enabled.
                version = get_next_version(
                    manifest,
                    key,
                    connection=manifest_connection,
                )

                # Collect folder tags from .lsm_tags.json files
                f_tags = collect_folder_tags(fp, root_cfg.path)

                # Submit parse/chunk to pool
                futures.append(
                    pool.submit(
                        parse_and_chunk_job,
                        fp,
                        source_path,
                        mtime_ns,
                        size,
                        fhash,
                        had_prev,
                        enable_ocr,
                        skip_errors,
                        stop_signal,
                        chunk_size,
                        chunk_overlap,
                        chunking_strategy,
                        enable_language_detection,
                        enable_translation,
                        translation_target,
                        translation_llm_config,
                        root_cfg.tags,
                        root_cfg.content_type,
                        f_tags or None,
                    )
                )
                # Store version on the future so embed worker can pick it up
                futures[-1]._lsm_version = version
                processed += 1
                files_submitted += 1

                # Check file count limit
                if max_files is not None and files_submitted >= max_files:
                    emit("limit", files_submitted, total_files,
                         f"Reached max_files limit ({max_files})")
                    limit_reached = True
                    break

                # Drain completed futures opportunistically to keep memory stable
                if len(futures) >= parse_workers * 4:
                    done = [f for f in futures if f.done()]
                    for f in done:
                        futures.remove(f)
                        pr = f.result()
                        pr.version = getattr(f, "_lsm_version", 1)
                        if pr.parse_errors:
                            error_records["page_errors"].extend(
                                {
                                    "source_path": pr.source_path,
                                    "page": err.get("page"),
                                    "stage": err.get("stage"),
                                    "error": err.get("error"),
                                }
                                for err in pr.parse_errors
                            )
                        if not pr.ok:
                            error_records["failed_documents"].append(
                                {
                                    "source_path": pr.source_path,
                                    "ext": pr.ext,
                                    "error": pr.err,
                                }
                            )
                            if not skip_errors:
                                raise RuntimeError(f"Parse failed for {pr.source_path}: {pr.err}")
                            skipped += 1
                        else:
                            parse_out_q.put(pr)
                        completed += 1
                    maybe_report()

            # Drain remaining parse futures
            for f in futures:
                if stop_signal.is_set():
                    break
                pr = f.result()
                pr.version = getattr(f, "_lsm_version", 1)
                if pr.parse_errors:
                    error_records["page_errors"].extend(
                        {
                            "source_path": pr.source_path,
                            "page": err.get("page"),
                            "stage": err.get("stage"),
                            "error": err.get("error"),
                        }
                        for err in pr.parse_errors
                    )
                if not pr.ok:
                    error_records["failed_documents"].append(
                        {
                            "source_path": pr.source_path,
                            "ext": pr.ext,
                            "error": pr.err,
                        }
                    )
                    if not skip_errors:
                        raise RuntimeError(f"Parse failed for {pr.source_path}: {pr.err}")
                    skipped += 1
                else:
                    parse_out_q.put(pr)
                completed += 1
                maybe_report()
    except KeyboardInterrupt:
        stop_signal.set()
        interrupted = True
    else:
        interrupted = False

    # Signal embedder shutdown
    parse_out_q.put(None)

    # Wait for threads
    et.join()
    wt.join()

    if writer_error is not None:
        raise RuntimeError(f"Ingest write stage failed: {writer_error}") from writer_error

    if interrupted:
        emit("interrupt", completed, total_files, "Stopping ingest (received interrupt)")

    if not dry_run:
        save_manifest(manifest_path, manifest, connection=manifest_connection)
        error_report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": now_iso(),
            "total_failed_documents": len(error_records["failed_documents"]),
            "total_page_errors": len(error_records["page_errors"]),
            "failed_documents": error_records["failed_documents"],
            "page_errors": error_records["page_errors"],
        }
        error_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    elapsed = time.time() - start_time
    emit(
        "complete",
        completed,
        total_files,
        (
            f"completed={completed} skipped={skipped} embedded_files={embedded_files} "
            f"chunks_added={added_chunks} chunks_written={written_chunks} "
            f"elapsed={format_time(elapsed)} vectordb={vectordb_config.provider} "
            f"collection={vectordb_config.collection} dry_run={dry_run}"
        ),
    )

    return {
        "total_files": total_files,
        "completed_files": completed,
        "skipped_files": skipped,
        "embedded_files": embedded_files,
        "chunks_added": added_chunks,
        "chunks_written": written_chunks,
        "elapsed_seconds": elapsed,
        "errors": error_records["failed_documents"],
        "page_errors": error_records["page_errors"],
        "interrupted": interrupted,
        "files_submitted": files_submitted,
        "limit_reached": limit_reached,
    }
