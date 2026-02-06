import pytest


def test_chromadb_module_importable() -> None:
    pytest.importorskip("chromadb")
    from lsm.vectordb.chromadb import ChromaDBProvider  # noqa: F401

