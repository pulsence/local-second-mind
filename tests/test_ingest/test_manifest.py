from lsm.ingest.manifest import load_manifest


def test_load_manifest_missing_file_returns_empty_dict(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert load_manifest(missing) == {}

