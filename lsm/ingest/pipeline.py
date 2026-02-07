from __future__ import annotations

import os
import time
import threading
import queue
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

from lsm.ingest.chunking import chunk_text
from lsm.ingest.fs import iter_files
from lsm.ingest.manifest import load_manifest, save_manifest
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
from lsm.vectordb import create_vectordb_provider
from lsm.config.models import LLMConfig, VectorDBConfig

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

def ingest(
    roots: List[Path],
    chroma_flush_interval: int,
    embed_model_name: str,
    device: str,
    batch_size: int,
    manifest_path: Path,
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
) -> Dict[str, Any]:
    """
    Run ingest pipeline.

    Returns:
        Summary dictionary with ingest metrics.
    """
    def emit(event: str, current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(event, current, total, message)

    stop_signal = stop_event or threading.Event()
    provider = create_vectordb_provider(vectordb_config)

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

    manifest = load_manifest(manifest_path)

    files = list(iter_files(roots, exts, exclude_dirs))
    total_files = len(files)
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
    written_chunks = 0
    error_report_path = manifest_path.parent / "ingest_error_report.json"
    error_records = {
        "failed_documents": [],
        "page_errors": [],
    }

    REPORT_EVERY_SECONDS = 1.0

    # ---- Writer thread (sole Chroma owner) ----
    def writer_thread():
        nonlocal written_chunks

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

            # Delete prior chunks only if the file previously existed in the manifest
            if job.had_prev:
                try:
                    provider.delete_by_filter({"source_path": job.source_path})
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

                to_add_ids.append(chunk_id)
                to_add_docs.append(chunk)
                to_add_metas.append(meta)
                to_add_embs.append(emb)

            # Stage manifest update for this file, but do NOT commit yet
            pending_manifest_updates[job.source_path] = {
                "mtime_ns": job.mtime_ns,
                "size": job.size,
                "file_hash": job.file_hash,
                "updated_at": now_iso(),
            }

            # Flush when we hit threshold
            if len(to_add_ids) >= chroma_flush_interval:
                flush()

        # Final flush
        flush()

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

    try:
        with ThreadPoolExecutor(max_workers=parse_workers) as pool:
            futures = []

            for fp in files:
                if stop_signal.is_set():
                    break
                scanned += 1
                source_path = canonical_path(fp)
                key = source_path

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

                # Fast skip on (mtime,size) + hash present
                if prev and prev.get("mtime_ns") == mtime_ns and prev.get("size") == size and prev.get("file_hash"):
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

                if prev and prev.get("file_hash") == fhash:
                    # content unchanged, update manifest
                    manifest[key] = {"mtime_ns": mtime_ns, "size": size, "file_hash": fhash, "updated_at": now_iso()}
                    skipped += 1
                    processed += 1
                    completed += 1
                    maybe_report()
                    continue

                had_prev = prev is not None

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
                    )
                )
                processed += 1

                # Drain completed futures opportunistically to keep memory stable
                if len(futures) >= parse_workers * 4:
                    done = [f for f in futures if f.done()]
                    for f in done:
                        futures.remove(f)
                        pr = f.result()
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

    if interrupted:
        emit("interrupt", completed, total_files, "Stopping ingest (received interrupt)")

    if not dry_run:
        save_manifest(manifest_path, manifest)
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
    }
