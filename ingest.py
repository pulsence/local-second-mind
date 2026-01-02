#!/usr/bin/env python3
"""
ingest.py — crawl + parse + chunk + embed + persist to Chroma (persistent)

Minimal, working skeleton for a heterogeneous folder tree on Windows.

Install (example):
  pip install chromadb sentence-transformers pypdf python-docx beautifulsoup4 lxml

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

# Local embeddings
from sentence_transformers import SentenceTransformer

# Parsers (optional by extension)
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup


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
    reader = PdfReader(str(path), strict=False)
    parts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            parts.append(t)
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
    Idempotent write:
    - Prefer upsert (overwrite existing IDs)
    - Fall back to add, and if add fails due to duplicates, delete IDs then add.
    """
    try:
        # Newer Chroma clients
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
        return
    except AttributeError:
        # Older clients: no upsert
        pass

    try:
        collection.add(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
    except Exception as e:
        # Best-effort recovery for DuplicateIDError / conflicts:
        try:
            collection.delete(ids=ids)
            collection.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
        except Exception:
            raise e


# -----------------------------
# Main ingest pipeline
# -----------------------------
def ingest(
    roots: List[Path],
    persist_dir: Path,
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

    to_add_ids: List[str] = []
    to_add_docs: List[str] = []
    to_add_metas: List[Dict] = []
    to_add_embs: List[List[float]] = []
    seen_ids: set[str] = set()
    visited_files: set[str] = set()

    start_time = time.time()
    last_report = start_time
    embed_seconds = 0.0
    REPORT_EVERY_SECONDS = 1.0

    files = list(iter_files(roots, exts, exclude_dirs))
    total_files = len(files)

    print(f"[INGEST] Discovered {total_files:,} files to consider")

    scanned = 0
    skipped = 0
    added_chunks = 0
    processed = 0
    embedded_files = 0 

    for fp in files:
        source_path = canonical_path(fp)

        fp_key = source_path
        if fp_key in visited_files:
            continue
        visited_files.add(fp_key)

        scanned += 1
        processed += 1
        key = source_path

        try:
            mtime_ns = fp.stat().st_mtime_ns
        except Exception:
            skipped += 1
            continue

        prev = manifest.get(key)
        # Fast skip on mtime if unchanged
        if prev and prev.get("mtime_ns") == mtime_ns and prev.get("file_hash"):
            skipped += 1
            continue

        # Hash file to detect change robustly
        try:
            fhash = file_sha256(fp)
        except Exception:
            skipped += 1
            continue

        if prev and prev.get("file_hash") == fhash:
            # mtime changed but content hash same
            manifest[key] = {"mtime_ns": mtime_ns, "file_hash": fhash, "updated_at": now_iso()}
            continue

        # Parse
        try:
            raw_text = parse_file(fp)
        except Exception as e:
            print(f"[WARN] parse failed: {fp} ({e})", file=sys.stderr)
            skipped += 1
            continue

        raw_text = normalize_whitespace(raw_text)
        if not raw_text:
            skipped += 1
            continue

        chunks = chunk_text(raw_text)
        if not chunks:
            skipped += 1
            continue

        embedded_files += 1

        # OPTIONAL: delete old chunks for prior versions of this file
        # We cannot easily query by metadata in all Chroma configs; simplest is:
        # - If you stored 'source_path' as metadata, you *can* try deleting by where filter.
        # If your chroma version supports it, this is recommended to prevent duplicates.
        try:
            collection.delete(where={"source_path": source_path})
        except Exception:
            # If delete(where=...) isn't supported in your environment, ignore.
            pass

        # Embed + stage for add
        # Batch embedding for speed
        t0 = time.time()
        embeddings = model.encode(chunks, batch_size=batch_size,
                                  show_progress_bar=False, normalize_embeddings=True)
        embed_seconds += (time.time() - t0)

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = make_chunk_id(source_path, fhash, idx)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = {
                "source_path": source_path,
                "source_name": fp.name,
                "ext": fp.suffix.lower(),
                "mtime_ns": mtime_ns,
                "file_hash": fhash,
                "chunk_index": idx,
                "ingested_at": now_iso(),
            }
            to_add_ids.append(cid)
            to_add_docs.append(chunk)
            to_add_metas.append(meta)
            to_add_embs.append(emb.tolist())

        manifest[key] = {"mtime_ns": mtime_ns, "file_hash": fhash, "updated_at": now_iso()}

        added_chunks += len(chunks)

        # Flush periodically to avoid huge RAM usage
        if len(to_add_ids) >= 2000:
            if not dry_run:
                chroma_write(collection,
                             ids=to_add_ids,
                             documents=to_add_docs,
                             metadatas=to_add_metas,
                             embeddings=to_add_embs)
            to_add_ids, to_add_docs, to_add_metas, to_add_embs = [], [], [], []
            seen_ids.clear()
        
        # Periodic progress report
        now = time.time()
        if (now - last_report) >= REPORT_EVERY_SECONDS:
            elapsed = now  - start_time
            rate = scanned / elapsed if elapsed > 0 else 0.0
            remaining = total_files - scanned

            chunk_rate = (added_chunks / embed_seconds) if embed_seconds > 0 else 0.0
            eta = (remaining / rate) if rate > 0 else float("inf")

            print(
                f"[INGEST] scanned={scanned:,}/{total_files:,} files  "
                f"processed={processed:,}  embedded={embedded_files:,}  "
                f"added chunks={added_chunks:,}  "
                f"skipped={skipped:,}  "
                f"rate={rate:0.2f} files/sec  "
                f"embed={chunk_rate:0.1f} chunks/sec  "
                f"elapsed={format_time(elapsed)}  ETA={format_time(eta)}"
            )
            last_report = now


    # Final flush
    if to_add_ids:
        if not dry_run:
            chroma_write(collection,
                         ids=to_add_ids,
                         documents=to_add_docs,
                         metadatas=to_add_metas,
                         embeddings=to_add_embs)

    if not dry_run:
        save_manifest(manifest_path, manifest)

    print(
        f"Done.\n"
        f"  scanned files:   {scanned}\n"
        f"  skipped files:   {skipped}\n"
        f"  added chunks:    {added_chunks}\n"
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

    logging.getLogger("pypdf").setLevel(logging.ERROR)

    print(f"[INGEST] Starting ingest with config:\n{cfg}")

    ingest(
        roots=cfg["roots"],
        persist_dir=cfg["persist_dir"],
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