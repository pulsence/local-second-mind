from lsm.ingest.manifest import load_manifest, save_manifest


def test_load_manifest_missing_file_returns_empty_dict(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    assert load_manifest(missing) == {}


def test_load_manifest_invalid_json_returns_empty_dict(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{invalid json", encoding="utf-8")
    assert load_manifest(manifest_path) == {}


def test_save_manifest_creates_parent_and_roundtrips(tmp_path) -> None:
    manifest_path = tmp_path / ".ingest" / "manifest.json"
    payload = {"doc.txt": {"hash": "abc123", "mtime_ns": 12345}}

    save_manifest(manifest_path, payload)

    assert manifest_path.exists()
    assert load_manifest(manifest_path) == payload
