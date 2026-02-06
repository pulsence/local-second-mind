from pathlib import Path

from lsm.config.models.ingest import IngestConfig


def test_ingest_config_converts_string_paths() -> None:
    cfg = IngestConfig(roots=["."], persist_dir=".chroma", manifest=".ingest/manifest.json")
    assert isinstance(cfg.roots[0], Path)
    assert isinstance(cfg.persist_dir, Path)
    assert isinstance(cfg.manifest, Path)

