from pathlib import Path

import pytest

from lsm.config.models.ingest import IngestConfig, RootConfig


def test_ingest_config_converts_string_paths() -> None:
    cfg = IngestConfig(roots=["."], persist_dir=".chroma", manifest=".ingest/manifest.json")
    assert isinstance(cfg.roots[0], RootConfig)
    assert cfg.roots[0].path == Path(".")
    assert isinstance(cfg.persist_dir, Path)
    assert isinstance(cfg.manifest, Path)


def test_language_detection_defaults_to_false() -> None:
    cfg = IngestConfig(roots=["/tmp"])
    assert cfg.enable_language_detection is False


def test_translation_defaults_to_false() -> None:
    cfg = IngestConfig(roots=["/tmp"])
    assert cfg.enable_translation is False


def test_translation_target_defaults_to_en() -> None:
    cfg = IngestConfig(roots=["/tmp"])
    assert cfg.translation_target == "en"


def test_translation_requires_language_detection() -> None:
    cfg = IngestConfig(
        roots=["/tmp"],
        enable_translation=True,
        enable_language_detection=False,
    )
    with pytest.raises(ValueError, match="enable_language_detection"):
        cfg.validate()


def test_translation_requires_target() -> None:
    cfg = IngestConfig(
        roots=["/tmp"],
        enable_translation=True,
        enable_language_detection=True,
        translation_target="",
    )
    with pytest.raises(ValueError, match="translation_target"):
        cfg.validate()


def test_translation_valid_config_accepted() -> None:
    cfg = IngestConfig(
        roots=["/tmp"],
        enable_translation=True,
        enable_language_detection=True,
        translation_target="en",
    )
    cfg.validate()  # Should not raise


def test_language_detection_without_translation_accepted() -> None:
    cfg = IngestConfig(
        roots=["/tmp"],
        enable_language_detection=True,
        enable_translation=False,
    )
    cfg.validate()  # Should not raise

