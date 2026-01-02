from __future__ import annotations

import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone

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