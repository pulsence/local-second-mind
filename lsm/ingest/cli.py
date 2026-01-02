from __future__ import annotations

from pathlib import Path

from lsm.ingest.config import load_config, normalize_config
from lsm.ingest.pipeline import ingest

def main(config_path: str) -> int:
    cfg_path = Path(config_path).expanduser().resolve()

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
