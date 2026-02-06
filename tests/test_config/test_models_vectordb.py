from pathlib import Path

from lsm.config.models.vectordb import VectorDBConfig


def test_vectordb_config_converts_persist_dir() -> None:
    cfg = VectorDBConfig(persist_dir=".chroma")
    assert isinstance(cfg.persist_dir, Path)

