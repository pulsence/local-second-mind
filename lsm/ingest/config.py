from __future__ import annotations

import json
import yaml
from typing import Dict
from pathlib import Path

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