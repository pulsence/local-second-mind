#!/usr/bin/env python3
"""
ingest.py — crawl + parse + chunk + embed + persist to Chroma (persistent)

Minimal, working skeleton for a heterogeneous folder tree on Windows.

Notes:
- This uses *local* embeddings via sentence-transformers (no API keys).
- Incremental ingest is supported via a simple manifest (hash-by-file + mtime).
"""

import argparse
import hashlib
import json
import os
import re
import sys
import yaml
import logging
import time

import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

# Local embeddings
from sentence_transformers import SentenceTransformer

# Parsers (optional by extension)
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup

# -----------------------------
# Threading Classes
# -----------------------------
@dataclass
class ParseResult:
    source_path: str
    fp: Path
    mtime_ns: int
    size: int
    file_hash: str
    chunks: List[str]
    ext: str
    had_prev: bool  # whether manifest had an entry (controls delete(where))
    ok: bool
    err: Optional[str] = None

@dataclass
class WriteJob:
    # One job corresponds to one file (so writer can delete/manifest-update per file)
    source_path: str
    fp: Path
    mtime_ns: int
    size: int
    file_hash: str
    ext: str
    chunks: List[str]
    embeddings: List[List[float]]
    had_prev: bool

# -----------------------------
# Config
# -----------------------------
DEFAULT_EXTS = {
    ".txt", ".md", ".rst",
    ".pdf",
    ".docx",
    ".html", ".htm",
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__",
    ".venv", "venv",
    "node_modules",
}

CHUNK_SIZE_CHARS = 1800      # minimal; character-based for simplicity
CHUNK_OVERLAP_CHARS = 200

DEFAULT_COLLECTION = "local_kb"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Utility helpers
# -----------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def normalize_whitespace(s: str) -> str:
    # Collapse excessive whitespace without destroying paragraph boundaries too aggressively
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def canonical_path(p: Path) -> str:
    # Normalize for Windows stability: absolute + resolved + lowercase
    return str(p.expanduser().resolve()).lower()

def format_time(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "Lost in space and time"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

# -----------------------------
# Crawl
# -----------------------------
def iter_files(
    roots: List[Path],
    exts: set[str],
    exclude_dirs: set[str],
) -> Iterable[Path]:
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            # Skip excluded dirs by checking any parent folder name
            if any(parent.name in exclude_dirs for parent in p.parents):
                continue
            if p.suffix.lower() in exts:
                yield p


# -----------------------------
# Parse (by extension)
# -----------------------------
def parse_txt(path: Path) -> str:
    # Try utf-8, then fallback
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")

def parse_pdf(path: Path) -> str:
    parts: List[str] = []
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                blocks = page.get_text("blocks") or []
                for b in blocks:
                    txt = b[4]
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt)
    except Exception:
        return ""
    return "\n\n".join(parts)

def parse_docx(path: Path) -> str:
    doc = Document(str(path))
    parts: List[str] = []
    for para in doc.paragraphs:
        if para.text:
            parts.append(para.text)
    return "\n".join(parts)

def parse_html(path: Path) -> str:
    raw = parse_txt(path)
    soup = BeautifulSoup(raw, "lxml")
    # Remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return text

def parse_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".rst"}:
        return parse_txt(path)
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)
    # Fallback best-effort
    return parse_txt(path)

# -----------------------------
# Chunk (simple, minimal)
# -----------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> List[str]:
    """
    Minimal chunker:
    - Normalize whitespace
    - Split into chunks by char count with overlap
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    chunks: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap)

    return chunks


# -----------------------------
# Manifest for incremental ingest
# -----------------------------
def load_manifest(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_manifest(path: Path, manifest: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------
# Chroma persistence
# -----------------------------
def get_chroma_collection(persist_dir: Path, name: str):
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=name,
        # cosine is typical for sentence-transformers embeddings
        metadata={"hnsw:space": "cosine"},
    )

def make_chunk_id(source_path: str, file_hash: str, chunk_index: int) -> str:
    """
    Generate a unique chunk id per (source file path, file version, chunk index).

    This prevents collisions when two different files have identical content
    (same file_hash), and avoids DuplicateIDError in Chroma.
    """
    sp_norm = source_path.lower()
    sp_hash = hashlib.sha256(sp_norm.encode("utf-8")).hexdigest()
    return f"{sp_hash}:{file_hash}:{chunk_index}"

def chroma_write(collection, ids, documents, metadatas, embeddings):
    """
    Idempotent write with automatic sub-batching to respect Chroma max batch size.
    - Prefer upsert (overwrite existing IDs)
    - Fall back to add (+ delete-then-add recovery)
    """
    if not ids:
        return

    # Determine the maximum batch size allowed by Chroma.
    # Newer chromadb exposes collection._client.get_max_batch_size()
    # Some versions expose it on collection._client, others on collection._client._server.
    max_bs = None
    try:
        max_bs = collection._client.get_max_batch_size()
    except Exception:
        pass

    # Fallback: probe via private attribute if present
    if max_bs is None:
        max_bs = getattr(collection, "max_batch_size", None)

    # Conservative fallback if we cannot introspect
    if not isinstance(max_bs, int) or max_bs <= 0:
        max_bs = 4000

    # Write in sub-batches
    n = len(ids)
    for i in range(0, n, max_bs):
        j = min(i + max_bs, n)
        sub_ids = ids[i:j]
        sub_docs = documents[i:j]
        sub_metas = metadatas[i:j]
        sub_embs = embeddings[i:j]

        # Prefer upsert if available
        try:
            collection.upsert(
                ids=sub_ids,
                documents=sub_docs,
                metadatas=sub_metas,
                embeddings=sub_embs,
            )
            continue
        except AttributeError:
            pass

        try:
            collection.add(
                ids=sub_ids,
                documents=sub_docs,
                metadatas=sub_metas,
                embeddings=sub_embs,
            )
        except Exception as e:
            # Best-effort recovery for duplicates/conflicts
            try:
                collection.delete(ids=sub_ids)
                collection.add(
                    ids=sub_ids,
                    documents=sub_docs,
                    metadatas=sub_metas,
                    embeddings=sub_embs,
                )
            except Exception:
                raise e

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
        nonlocal written_chunks, added_chunks, embedded_files

        to_add_ids: List[str] = []
        to_add_docs: List[str] = []
        to_add_metas: List[Dict] = []
        to_add_embs: List[List[float]] = []
        seen_ids: set[str] = set()

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

            # Update manifest only after staging (write will happen soon)
            # We update again after the final successful flush below.
            # (If you want strict "only after persisted", move this into flush-success branch.)
            manifest[job.source_path] = {
                "mtime_ns": job.mtime_ns,
                "size": job.size,
                "file_hash": job.file_hash,
                "updated_at": now_iso(),
            }

            # Flush
            if len(to_add_ids) >= chroma_flush_interval:
                if not dry_run:
                    chroma_write(collection, to_add_ids, to_add_docs, to_add_metas, to_add_embs)

                with lock:
                    written_chunks += len(to_add_ids)

                to_add_ids, to_add_docs, to_add_metas, to_add_embs = [], [], [], []
                seen_ids.clear()

        # Final flush
        if to_add_ids:
            if not dry_run:
                chroma_write(collection, to_add_ids, to_add_docs, to_add_metas, to_add_embs)
            with lock:
                written_chunks += len(to_add_ids)

    wt = threading.Thread(target=writer_thread, daemon=True)
    wt.start()

    # ---- Embed worker (single GPU consumer; batches across files) ----
    def embed_worker():
        nonlocal embed_seconds, embedded_files, added_chunks

        pending: List[ParseResult] = []
        pending_chunk_count = 0
        MAX_CHUNKS_PER_GPU_BATCH = max(batch_size * 16, 1024)  # tune: larger => better GPU utilization

        def flush_pending():
            nonlocal embed_seconds, embedded_files, added_chunks
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

# -----------------------------
# Config loading
# -----------------------------
def load_config(path: Path) -> Dict:
    """
    Load YAML or JSON config file.
    - .yaml / .yml => YAML
    - .json        => JSON
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw) or {}
    if suffix == ".json":
        return json.loads(raw)
    raise ValueError("Config must be .yaml/.yml or .json")


def normalize_config(cfg: Dict, cfg_path: Path) -> Dict:
    """
    Apply defaults and normalize types/values.
    Expected keys:
      roots: [str, ...]                (required)
      persist_dir: str                 (default ".chroma")
      chroma_flush_interval: int       (default 2000)
      collection: str                  (default DEFAULT_COLLECTION)
      embed_model: str                 (default DEFAULT_EMBED_MODEL)
      device: str                      (default "cpu")
      batch_size: int                  (default 32)
      manifest: str                    (default ".ingest/manifest.json")
      extensions: [".txt", ...]        (optional; merged with DEFAULT_EXTS unless override_extensions=true)
      override_extensions: bool        (default false)
      exclude_dirs: ["node_modules"]   (optional; merged with DEFAULT_EXCLUDE_DIRS unless override_excludes=true)
      override_excludes: bool          (default false)
      dry_run: bool                    (default false)
    """
    out: Dict = {}

    roots = cfg.get("roots")
    if not roots or not isinstance(roots, list):
        raise ValueError("Config must include 'roots' as a non-empty list of folder paths.")

    out["roots"] = [Path(r) for r in roots]
    out["collection"] = str(cfg.get("collection", DEFAULT_COLLECTION))
    out["chroma_flush_interval"] = int(cfg.get("chroma_flush_interval", 2000))

    out["embed_model"] = str(cfg.get("embed_model", DEFAULT_EMBED_MODEL))
    out["dry_run"] = bool(cfg.get("dry_run", False))
    out["device"] = str(cfg.get("device", "cpu"))
    out["batch_size"] = int(cfg.get("batch_size", 32))

    persist_raw = cfg.get("persist_dir", ".chroma")
    out["persist_dir"] = (cfg_path.parent / persist_raw).resolve()

    manifest_raw = cfg.get("manifest", ".ingest/manifest.json")
    out["manifest"] = (cfg_path.parent / manifest_raw).resolve()

    # Extensions
    cfg_exts = cfg.get("extensions", [])
    if cfg_exts and not isinstance(cfg_exts, list):
        raise ValueError("'extensions' must be a list (e.g., ['.txt', '.pdf']).")
    override_exts = bool(cfg.get("override_extensions", False))

    exts = set()
    if not override_exts:
        exts |= set(DEFAULT_EXTS)
    for e in cfg_exts:
        e = str(e).strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        exts.add(e.lower())
    out["exts"] = exts

    # Excludes
    cfg_excl = cfg.get("exclude_dirs", [])
    if cfg_excl and not isinstance(cfg_excl, list):
        raise ValueError("'exclude_dirs' must be a list (e.g., ['.git', 'node_modules']).")
    override_excl = bool(cfg.get("override_excludes", False))

    exclude_dirs = set()
    if not override_excl:
        exclude_dirs |= set(DEFAULT_EXCLUDE_DIRS)
    exclude_dirs |= {str(d) for d in cfg_excl if str(d).strip()}
    out["exclude_dirs"] = exclude_dirs

    return out


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[List[str]] = None) -> int:
    """
    Now reads a config file instead of individual CLI params.

    Usage:
      python config.py --config config.yaml
      python config.py --config config.json
    """
    ap = argparse.ArgumentParser(description="Config file specifying the local files into a persistent Chroma collection.")
    ap.add_argument(
        "--config",
        default="config.json",
        help="Path to YAML/JSON config file (e.g., config.yaml).",
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Ingest config not found: {cfg_path}\n"
            f"Either create it or pass --config explicitly."
        )

    cfg = load_config(cfg_path)
    cfg = normalize_config(cfg, cfg_path)

    print(f"[INGEST] Starting ingest with config:\n{cfg}")

    ingest(
        roots=cfg["roots"],
        persist_dir=cfg["persist_dir"],
        chroma_flush_interval=cfg["chroma_flush_interval"],
        collection_name=cfg["collection"],
        embed_model_name=cfg["embed_model"],
        device=cfg["device"],
        batch_size=cfg["batch_size"],
        manifest_path=cfg["manifest"],
        exts=cfg["exts"],
        exclude_dirs=cfg["exclude_dirs"],
        dry_run=cfg["dry_run"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())