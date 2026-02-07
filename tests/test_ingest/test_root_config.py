"""Tests for RootConfig, IngestConfig root parsing, folder tag discovery, and metadata propagation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from lsm.config.models.ingest import IngestConfig, RootConfig


# ---- RootConfig dataclass tests ----


class TestRootConfig:
    def test_path_conversion_from_string(self) -> None:
        rc = RootConfig(path="/tmp/docs")
        assert isinstance(rc.path, Path)
        assert rc.path == Path("/tmp/docs")

    def test_path_stays_path(self) -> None:
        rc = RootConfig(path=Path("/tmp/docs"))
        assert isinstance(rc.path, Path)

    def test_defaults_none(self) -> None:
        rc = RootConfig(path="/tmp")
        assert rc.tags is None
        assert rc.content_type is None

    def test_with_tags(self) -> None:
        rc = RootConfig(path="/tmp", tags=["research", "papers"])
        assert rc.tags == ["research", "papers"]

    def test_with_content_type(self) -> None:
        rc = RootConfig(path="/tmp", content_type="writing")
        assert rc.content_type == "writing"

    def test_with_all_fields(self) -> None:
        rc = RootConfig(path="/tmp", tags=["a", "b"], content_type="notes")
        assert rc.path == Path("/tmp")
        assert rc.tags == ["a", "b"]
        assert rc.content_type == "notes"


# ---- IngestConfig roots backward compatibility ----


class TestIngestConfigRoots:
    def test_string_roots_converted_to_root_config(self) -> None:
        cfg = IngestConfig(roots=["/tmp/docs", "/tmp/notes"])
        assert len(cfg.roots) == 2
        for rc in cfg.roots:
            assert isinstance(rc, RootConfig)
        assert cfg.roots[0].path == Path("/tmp/docs")
        assert cfg.roots[1].path == Path("/tmp/notes")
        assert cfg.roots[0].tags is None
        assert cfg.roots[0].content_type is None

    def test_path_roots_converted_to_root_config(self) -> None:
        cfg = IngestConfig(roots=[Path("/tmp/docs")])
        assert isinstance(cfg.roots[0], RootConfig)
        assert cfg.roots[0].path == Path("/tmp/docs")

    def test_dict_roots_converted_to_root_config(self) -> None:
        cfg = IngestConfig(
            roots=[
                {"path": "/tmp/docs", "tags": ["research"], "content_type": "academic"}
            ]
        )
        assert isinstance(cfg.roots[0], RootConfig)
        assert cfg.roots[0].path == Path("/tmp/docs")
        assert cfg.roots[0].tags == ["research"]
        assert cfg.roots[0].content_type == "academic"

    def test_dict_roots_without_optional_fields(self) -> None:
        cfg = IngestConfig(roots=[{"path": "/tmp/docs"}])
        assert isinstance(cfg.roots[0], RootConfig)
        assert cfg.roots[0].tags is None
        assert cfg.roots[0].content_type is None

    def test_root_config_objects_kept_as_is(self) -> None:
        rc = RootConfig(path="/tmp/docs", tags=["x"])
        cfg = IngestConfig(roots=[rc])
        assert cfg.roots[0] is rc

    def test_mixed_roots(self) -> None:
        cfg = IngestConfig(
            roots=[
                "/tmp/plain",
                Path("/tmp/pathobj"),
                {"path": "/tmp/tagged", "tags": ["t1"]},
                RootConfig(path="/tmp/rc", content_type="ct"),
            ]
        )
        assert len(cfg.roots) == 4
        for rc in cfg.roots:
            assert isinstance(rc, RootConfig)
        assert cfg.roots[0].path == Path("/tmp/plain")
        assert cfg.roots[1].path == Path("/tmp/pathobj")
        assert cfg.roots[2].tags == ["t1"]
        assert cfg.roots[3].content_type == "ct"

    def test_root_paths_property(self) -> None:
        cfg = IngestConfig(
            roots=[
                "/tmp/a",
                {"path": "/tmp/b", "tags": ["x"]},
            ]
        )
        paths = cfg.root_paths
        assert paths == [Path("/tmp/a"), Path("/tmp/b")]
        for p in paths:
            assert isinstance(p, Path)

    def test_validate_requires_roots(self) -> None:
        cfg = IngestConfig(roots=[])
        with pytest.raises(ValueError, match="root directory"):
            cfg.validate()


# ---- Config loader tests ----


class TestConfigLoaderRoots:
    def _base_raw(self, tmp_path: Path) -> Dict[str, Any]:
        return {
            "global": {
                "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
            },
            "ingest": {
                "roots": [str(tmp_path / "docs")],
                "persist_dir": str(tmp_path / ".chroma"),
                "collection": "test_kb",
                "manifest": str(tmp_path / ".ingest" / "manifest.json"),
            },
            "vectordb": {
                "provider": "chromadb",
                "persist_dir": str(tmp_path / ".chroma"),
                "collection": "test_kb",
            },
            "llms": {
                "providers": [{"provider_name": "openai", "api_key": None}],
                "services": {
                    "query": {"provider": "openai", "model": "gpt-5.2"}
                },
            },
            "query": {"k": 12, "k_rerank": 6},
        }

    def test_build_ingest_config_string_roots(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = self._base_raw(tmp_path)
        config = build_ingest_config(raw, tmp_path / "config.json")
        assert len(config.roots) == 1
        assert isinstance(config.roots[0], RootConfig)
        assert config.roots[0].path == Path(tmp_path / "docs")
        assert config.roots[0].tags is None

    def test_build_ingest_config_dict_roots(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = self._base_raw(tmp_path)
        raw["ingest"]["roots"] = [
            {"path": str(tmp_path / "docs"), "tags": ["research"], "content_type": "academic"}
        ]
        config = build_ingest_config(raw, tmp_path / "config.json")
        assert config.roots[0].tags == ["research"]
        assert config.roots[0].content_type == "academic"

    def test_build_ingest_config_mixed_roots(self, tmp_path: Path) -> None:
        from lsm.config.loader import build_ingest_config

        raw = self._base_raw(tmp_path)
        raw["ingest"]["roots"] = [
            str(tmp_path / "plain"),
            {"path": str(tmp_path / "tagged"), "tags": ["t1"]},
        ]
        config = build_ingest_config(raw, tmp_path / "config.json")
        assert len(config.roots) == 2
        assert config.roots[0].tags is None
        assert config.roots[1].tags == ["t1"]

    def _make_config(self, tmp_path: Path, roots: list) -> "LSMConfig":
        """Helper to build an LSMConfig with given roots (avoids API key validation)."""
        from lsm.config.models import (
            LSMConfig, GlobalConfig, QueryConfig, VectorDBConfig,
            LLMRegistryConfig, LLMProviderConfig, LLMServiceConfig,
        )

        return LSMConfig(
            ingest=IngestConfig(
                roots=roots,
                persist_dir=tmp_path / ".chroma",
                collection="test_kb",
                manifest=tmp_path / ".ingest" / "manifest.json",
            ),
            query=QueryConfig(),
            llm=LLMRegistryConfig(
                providers=[LLMProviderConfig(provider_name="local")],
                services={"default": LLMServiceConfig(provider="local", model="test")},
            ),
            vectordb=VectorDBConfig(
                provider="chromadb",
                persist_dir=tmp_path / ".chroma",
                collection="test_kb",
            ),
            global_settings=GlobalConfig(),
        )

    def test_config_to_raw_plain_roots(self, tmp_path: Path) -> None:
        from lsm.config.loader import config_to_raw

        config = self._make_config(tmp_path, [str(tmp_path / "docs")])
        out = config_to_raw(config)
        roots_out = out["ingest"]["roots"]
        assert len(roots_out) == 1
        assert isinstance(roots_out[0], str)

    def test_config_to_raw_tagged_roots(self, tmp_path: Path) -> None:
        from lsm.config.loader import config_to_raw

        config = self._make_config(tmp_path, [
            str(tmp_path / "plain"),
            RootConfig(path=tmp_path / "tagged", tags=["t1"], content_type="notes"),
        ])
        out = config_to_raw(config)
        roots_out = out["ingest"]["roots"]
        assert isinstance(roots_out[0], str)  # plain root
        assert isinstance(roots_out[1], dict)  # tagged root
        assert roots_out[1]["tags"] == ["t1"]
        assert roots_out[1]["content_type"] == "notes"

    def test_config_to_raw_roundtrip(self, tmp_path: Path) -> None:
        from lsm.config.loader import config_to_raw, build_ingest_config

        config = self._make_config(tmp_path, [
            str(tmp_path / "a"),
            RootConfig(path=tmp_path / "b", tags=["x", "y"], content_type="ct"),
        ])
        out = config_to_raw(config)
        # Re-parse ingest section from serialized output
        ingest2 = build_ingest_config(out, tmp_path / "config.json")
        assert len(ingest2.roots) == 2
        assert ingest2.roots[1].tags == ["x", "y"]
        assert ingest2.roots[1].content_type == "ct"


# ---- Folder tag discovery tests ----


class TestFolderTagDiscovery:
    def test_collect_folder_tags_no_tag_files(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        root.mkdir()
        sub = root / "sub"
        sub.mkdir()
        file_path = sub / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        assert tags == []

    def test_collect_folder_tags_in_root(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        root.mkdir()
        (root / ".lsm_tags.json").write_text(json.dumps({"tags": ["root_tag"]}))
        file_path = root / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        assert tags == ["root_tag"]

    def test_collect_folder_tags_in_subfolder(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (sub / ".lsm_tags.json").write_text(json.dumps({"tags": ["sub_tag"]}))
        file_path = sub / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        assert tags == ["sub_tag"]

    def test_collect_folder_tags_nested_accumulates(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        sub = root / "a" / "b"
        sub.mkdir(parents=True)
        (root / ".lsm_tags.json").write_text(json.dumps({"tags": ["root_tag"]}))
        (root / "a" / ".lsm_tags.json").write_text(json.dumps({"tags": ["a_tag"]}))
        (sub / ".lsm_tags.json").write_text(json.dumps({"tags": ["b_tag"]}))
        file_path = sub / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        # Root-level first, most specific last
        assert tags == ["root_tag", "a_tag", "b_tag"]

    def test_collect_folder_tags_invalid_json_skipped(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        root.mkdir()
        (root / ".lsm_tags.json").write_text("not valid json {{{")
        file_path = root / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        assert tags == []

    def test_collect_folder_tags_missing_tags_key(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        root.mkdir()
        (root / ".lsm_tags.json").write_text(json.dumps({"other": "data"}))
        file_path = root / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        assert tags == []

    def test_collect_folder_tags_deduplicates(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import collect_folder_tags

        root = tmp_path / "root"
        sub = root / "sub"
        sub.mkdir(parents=True)
        (root / ".lsm_tags.json").write_text(json.dumps({"tags": ["shared", "root_only"]}))
        (sub / ".lsm_tags.json").write_text(json.dumps({"tags": ["shared", "sub_only"]}))
        file_path = sub / "test.txt"
        file_path.write_text("content")

        tags = collect_folder_tags(file_path, root)
        # "shared" should only appear once
        assert tags.count("shared") == 1
        assert "root_only" in tags
        assert "sub_only" in tags


# ---- iter_files tests ----


class TestIterFilesWithRootConfig:
    def test_iter_files_returns_tuples(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import iter_files

        root = tmp_path / "docs"
        root.mkdir()
        (root / "file.txt").write_text("hello")
        rc = RootConfig(path=root)

        results = list(iter_files([rc], {".txt"}, set()))
        assert len(results) == 1
        file_path, root_config = results[0]
        assert isinstance(file_path, Path)
        assert isinstance(root_config, RootConfig)
        assert root_config is rc

    def test_iter_files_with_tagged_roots(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import iter_files

        root1 = tmp_path / "research"
        root1.mkdir()
        (root1 / "paper.txt").write_text("paper")
        rc1 = RootConfig(path=root1, tags=["research"], content_type="academic")

        root2 = tmp_path / "notes"
        root2.mkdir()
        (root2 / "note.txt").write_text("note")
        rc2 = RootConfig(path=root2, tags=["personal"])

        results = list(iter_files([rc1, rc2], {".txt"}, set()))
        assert len(results) == 2

        # Map file names to their root configs
        by_name = {fp.name: rc for fp, rc in results}
        assert by_name["paper.txt"].tags == ["research"]
        assert by_name["paper.txt"].content_type == "academic"
        assert by_name["note.txt"].tags == ["personal"]

    def test_iter_files_excludes_dirs(self, tmp_path: Path) -> None:
        from lsm.ingest.fs import iter_files

        root = tmp_path / "docs"
        (root / "cache").mkdir(parents=True)
        (root / "cache" / "cached.txt").write_text("cached")
        (root / "good.txt").write_text("good")
        rc = RootConfig(path=root)

        results = list(iter_files([rc], {".txt"}, {"cache"}))
        file_names = [fp.name for fp, _ in results]
        assert "good.txt" in file_names
        assert "cached.txt" not in file_names
