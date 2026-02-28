from pathlib import Path

from lsm.config.models.vectordb import VectorDBConfig


def test_vectordb_config_converts_path() -> None:
    cfg = VectorDBConfig(path="data")
    assert isinstance(cfg.path, Path)

