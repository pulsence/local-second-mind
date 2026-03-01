from pathlib import Path

from lsm.config.models import DBConfig


def test_vectordb_config_converts_path() -> None:
    cfg = DBConfig(path="data")
    assert isinstance(cfg.path, Path)

