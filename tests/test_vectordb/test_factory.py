from lsm.vectordb.factory import list_available_providers


def test_list_available_providers_contains_chromadb() -> None:
    assert "chromadb" in list_available_providers()

