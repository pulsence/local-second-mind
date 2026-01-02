from __future__ import annotations

import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer

from lsm.ingest.chunking import chunk_text
from lsm.ingest.chroma_store import chroma_write, get_chroma_collection, make_chunk_id
from lsm.ingest.fs import iter_files
from lsm.ingest.manifest import load_manifest, save_manifest
from lsm.ingest.models import ParseResult, WriteJob
from lsm.ingest.parsers import parse_file
from lsm.ingest.utils import canonical_path, file_sha256, now_iso, normalize_whitespace, format_time

# -----------------------------
# Main ingest pipeline
# -----------------------------
def parse_and_chunk_job(fp: Path, source_path: str, mtime_ns: int, size: int, fhash: str, had_prev: bool) -> ParseResult:
    try:
        raw_text = parse_file(fp)
        raw_text = normalize_whitespace(raw_text)
        if not raw_text:
            return ParseResult(source_path, fp, mtime_ns, size, fhash, [], fp.suffix.lower(), had_prev, ok=False, err="empty_text")

        chunks = chunk_text(raw_text)
        if not chunks:
            return ParseResult(source_path, fp, mtime_ns, size, fhash, [], fp.suffix.lower(), had_prev, ok=False, err="no_chunks")

        return ParseResult(source_path, fp, mtime_ns, size, fhash, chunks, fp.suffix.lower(), had_prev, ok=True)

    except Exception as e:
        return ParseResult(source_path, fp, mtime_ns, size, fhash, [], fp.suffix.lower(), had_prev, ok=False, err=str(e))

def ingest(
    roots: List[Path],
    persist_dir: Path,
    chroma_flush_interval: int,
    collection_name: str,
    embed_model_name: str,
    device: str,
    batch_size: int,
    manifest_path: Path,
    exts: set[str],
    exclude_dirs: set[str],
    dry_run: bool = False,
) -> None:
    collection = get_chroma_collection(persist_dir, collection_name)

    print(f"[INGEST] model device = {device}")
    model = SentenceTransformer(embed_model_name, device=device)

    manifest = load_manifest(manifest_path)

    files = list(iter_files(roots, exts, exclude_dirs))
    total_files = len(files)
    print(f"[INGEST] Discovered {total_files:,} files to consider")

    # ---- Queues (bounded for backpressure) ----
    parse_out_q: "queue.Queue[Optional[ParseResult]]" = queue.Queue(maxsize=128)
    write_q: "queue.Queue[Optional[WriteJob]]" = queue.Queue(maxsize=64)

    # ---- Metrics (thread-safe) ----
    lock = threading.Lock()
    start_time = time.time()
    last_report = start_time

    scanned = 0
    processed = 0
    skipped = 0
    embedded_files = 0
    added_chunks = 0
    embed_seconds = 0.0
    written_chunks = 0

    REPORT_EVERY_SECONDS = 1.0

    # ---- Writer thread (sole Chroma owner) ----
    def writer_thread():
        nonlocal written_chunks

        to_add_ids: List[str] = []
        to_add_docs: List[str] = []
        to_add_metas: List[Dict] = []
        to_add_embs: List[List[float]] = []
        seen_ids: set[str] = set()

        # Manifest updates are committed only after a successful chroma_write flush.
        pending_manifest_updates: Dict[str, Dict] = {}

        def flush():
            """Flush staged vectors to Chroma and, only on success, commit pending manifest updates."""
            nonlocal to_add_ids, to_add_docs, to_add_metas, to_add_embs, seen_ids, pending_manifest_updates, written_chunks

            if not to_add_ids:
                return

            if not dry_run:
                # If this raises, we do NOT update the manifest.
                chroma_write(collection, to_add_ids, to_add_docs, to_add_metas, to_add_embs)

            # Commit manifest updates only after successful write (or dry_run)
            manifest.update(pending_manifest_updates)
            pending_manifest_updates.clear()

            with lock:
                written_chunks += len(to_add_ids)

            to_add_ids, to_add_docs, to_add_metas, to_add_embs = [], [], [], []
            seen_ids.clear()

        while True:
            job = write_q.get()
            if job is None:
                break

            # Delete prior chunks only if the file previously existed in the manifest
            if job.had_prev:
                try:
                    collection.delete(where={"source_path": job.source_path})
                except Exception:
                    pass

            ingested_at = now_iso()

            # Stage chunks
            for idx, (chunk, emb) in enumerate(zip(job.chunks, job.embeddings)):
                chunk_id = make_chunk_id(job.source_path, job.file_hash, idx)
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)

                meta = {
                    "source_path": job.source_path,
                    "source_name": job.fp.name,
                    "ext": job.ext,
                    "mtime_ns": job.mtime_ns,
                    "file_hash": job.file_hash,
                    "chunk_index": idx,
                    "ingested_at": ingested_at,
                }

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
                )
                write_q.put(wj)

                with lock:
                    embedded_files += 1
                    added_chunks += len(pr.chunks)

            pending.clear()
            pending_chunk_count = 0 

        while True:
            pr = parse_out_q.get()
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
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = total_files - processed
            eta = (remaining / rate) if rate > 0 else float("inf")
            chunk_rate = (added_chunks / embed_seconds) if embed_seconds > 0 else 0.0
            wc = written_chunks

        print(
            f"[INGEST] processed={processed:,}/{total_files:,}  "
            f"skipped={skipped:,}  "
            f"embedded files={embedded_files:,}  "
            f"chunks added={added_chunks:,}  "
            f"chunks written={wc:,}  "
            f"file rate={rate:0.2f} files/sec  "
            f"embed={chunk_rate:0.1f} chunks/sec  "
            f"elapsed={format_time(elapsed)}  ETA={format_time(eta)}"
        )
        last_report = now

    with ThreadPoolExecutor(max_workers=parse_workers) as pool:
        futures = []

        for fp in files:
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
                continue

            prev = manifest.get(key)

            # Fast skip on (mtime,size) + hash present
            if prev and prev.get("mtime_ns") == mtime_ns and prev.get("size") == size and prev.get("file_hash"):
                skipped += 1
                processed += 1
                maybe_report()
                continue

            # Hash only when needed
            try:
                fhash = file_sha256(fp)
            except Exception:
                skipped += 1
                processed += 1
                maybe_report()
                continue

            if prev and prev.get("file_hash") == fhash:
                # content unchanged, update manifest
                manifest[key] = {"mtime_ns": mtime_ns, "size": size, "file_hash": fhash, "updated_at": now_iso()}
                skipped += 1
                processed += 1
                maybe_report()
                continue

            had_prev = prev is not None

            # Submit parse/chunk to pool
            futures.append(pool.submit(parse_and_chunk_job, fp, source_path, mtime_ns, size, fhash, had_prev))
            processed += 1

            # Drain completed futures opportunistically to keep memory stable
            if len(futures) >= parse_workers * 4:
                done = [f for f in futures if f.done()]
                for f in done:
                    futures.remove(f)
                    pr = f.result()
                    if not pr.ok:
                        skipped += 1
                    else:
                        parse_out_q.put(pr)
                maybe_report()

        # Drain remaining parse futures
        for f in futures:
            pr = f.result()
            if not pr.ok:
                skipped += 1
            else:
                parse_out_q.put(pr)
            maybe_report()

    # Signal embedder shutdown
    parse_out_q.put(None)

    # Wait for threads
    et.join()
    wt.join()

    if not dry_run:
        save_manifest(manifest_path, manifest)

    elapsed = time.time() - start_time
    print(
        f"Done.\n"
        f"  total files:     {total_files}\n"
        f"  processed files: {processed}\n"
        f"  skipped files:   {skipped}\n"
        f"  embedded files:  {embedded_files}\n"
        f"  chunks added:    {added_chunks}\n"
        f"  chunks written:  {written_chunks}\n"
        f"  elapsed:         {format_time(elapsed)}\n"
        f"  chroma dir:      {persist_dir}\n"
        f"  collection:      {collection_name}\n"
        f"  manifest:        {manifest_path}\n"
        f"  dry_run:         {dry_run}"
    )
