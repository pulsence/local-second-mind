from lsm.vectordb.utils import require_chroma_collection


def test_require_chroma_collection_passthrough_object() -> None:
    obj = object()
    assert require_chroma_collection(obj, "/test") is obj

