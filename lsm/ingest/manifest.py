from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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