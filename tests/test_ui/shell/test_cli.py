from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.config.models import VectorDBConfig
import lsm.ui.shell.cli as shell_cli


def test_run_ingest_dispatches_and_missing_subcommand(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(shell_cli, "run_build_cli", lambda *args, **kwargs: 11)
    monkeypatch.setattr(shell_cli, "run_tag_cli", lambda *args, **kwargs: 12)
    monkeypatch.setattr(shell_cli, "run_wipe_cli", lambda *args, **kwargs: 13)

    assert shell_cli.run_ingest(SimpleNamespace(ingest_command="build", config="c")) == 11
    assert shell_cli.run_ingest(SimpleNamespace(ingest_command="tag", config="c", max=1)) == 12
    assert shell_cli.run_ingest(SimpleNamespace(ingest_command="wipe", config="c", confirm=True)) == 13

    code = shell_cli.run_ingest(SimpleNamespace(ingest_command=None, config="c"))
    out = capsys.readouterr().out
    assert code == 2
    assert "Missing ingest subcommand" in out


def test_run_db_dispatches_and_missing_subcommand(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(shell_cli, "run_db_prune_cli", lambda *args, **kwargs: 21)
    monkeypatch.setattr(shell_cli, "run_db_complete_cli", lambda *args, **kwargs: 22)

    assert shell_cli.run_db(
        SimpleNamespace(db_command="prune", config="c", max_versions=3, older_than_days=7)
    ) == 21
    assert shell_cli.run_db(
        SimpleNamespace(db_command="complete", config="c", force_file_pattern="*.md")
    ) == 22

    code = shell_cli.run_db(SimpleNamespace(db_command=None, config="c"))
    out = capsys.readouterr().out
    assert code == 2
    assert "Missing db subcommand" in out


def test_run_migrate_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_cli, "run_migrate_cli", lambda *args, **kwargs: 31)
    code = shell_cli.run_migrate(
        SimpleNamespace(
            config="c",
            migration_source="sqlite",
            migration_target="sqlite",
            source_path=None,
            source_collection=None,
            source_connection_string=None,
            source_dir=None,
            target_path=None,
            target_collection=None,
            target_connection_string=None,
            batch_size=1000,
        )
    )
    assert code == 31


def test_load_config_missing_and_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        shell_cli._load_config(missing)

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    marker = object()
    monkeypatch.setattr(shell_cli, "load_config_from_file", lambda p: marker)
    assert shell_cli._load_config(cfg_file) is marker


def test_run_build_cli_overrides_and_progress(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = SimpleNamespace(ingest=SimpleNamespace(skip_errors=True, dry_run=False))
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)

    def _api_run_ingest(config, force=False, progress_callback=None, **kwargs):
        _ = kwargs
        progress_callback("progress", 1, 2, "halfway")
        progress_callback("done", 0, 0, "done")

    monkeypatch.setattr(shell_cli, "api_run_ingest", _api_run_ingest)
    code = shell_cli.run_build_cli("config.json", force=True, skip_errors=False, dry_run=True)
    out = capsys.readouterr().out

    assert code == 0
    assert cfg.ingest.skip_errors is False
    assert cfg.ingest.dry_run is True
    assert "[progress] 1/2 halfway" in out
    assert "[done] done" in out


def test_run_tag_cli(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = SimpleNamespace(
        vectordb=SimpleNamespace(provider="chromadb"),
        llm=SimpleNamespace(
            get_tagging_config=lambda: SimpleNamespace(model="gpt-test", provider="openai")
        ),
    )
    provider = object()
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)
    monkeypatch.setattr(shell_cli, "create_vectordb_provider", lambda _v: provider)
    monkeypatch.setattr(shell_cli, "tag_chunks", lambda **kwargs: (9, 2))

    code = shell_cli.run_tag_cli("config.json", max_chunks=10)
    out = capsys.readouterr().out
    assert code == 0
    assert "Starting AI tagging" in out
    assert "Successfully tagged: 9 chunks" in out


def test_run_wipe_cli_confirmation_and_error_paths(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code = shell_cli.run_wipe_cli("config.json", confirm=False)
    out = capsys.readouterr().out
    assert code == 2
    assert "Refusing to wipe" in out

    cfg = SimpleNamespace(collection="kb")
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)
    monkeypatch.setattr(shell_cli, "api_wipe_collection", lambda c: 5)
    code2 = shell_cli.run_wipe_cli("config.json", confirm=True)
    out2 = capsys.readouterr().out
    assert code2 == 0
    assert "Deleted 5 chunks" in out2

    def _boom(_cfg):
        raise RuntimeError("wipe failed")

    monkeypatch.setattr(shell_cli, "api_wipe_collection", _boom)
    code3 = shell_cli.run_wipe_cli("config.json", confirm=True)
    out3 = capsys.readouterr().out
    assert code3 == 1
    assert "Error: wipe failed" in out3


def test_run_db_prune_cli_success_and_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = SimpleNamespace(vectordb=SimpleNamespace(provider="sqlite"))
    provider = SimpleNamespace(prune_old_versions=lambda criteria: 7)
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)
    monkeypatch.setattr(shell_cli, "create_vectordb_provider", lambda _v: provider)

    code = shell_cli.run_db_prune_cli("config.json", max_versions=2, older_than_days=14)
    out = capsys.readouterr().out
    assert code == 0
    assert "Prune complete" in out

    def _boom(_criteria):
        raise RuntimeError("prune failed")

    provider_bad = SimpleNamespace(prune_old_versions=_boom)
    monkeypatch.setattr(shell_cli, "create_vectordb_provider", lambda _v: provider_bad)
    code2 = shell_cli.run_db_prune_cli("config.json")
    out2 = capsys.readouterr().out
    assert code2 == 1
    assert "Error: prune failed" in out2


def test_run_db_complete_cli_success_and_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg = SimpleNamespace()
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)

    def _ok_api_run_ingest(config, **kwargs):
        _ = config, kwargs
        return None

    monkeypatch.setattr(shell_cli, "api_run_ingest", _ok_api_run_ingest)
    code = shell_cli.run_db_complete_cli("config.json", force_file_pattern="*.md")
    out = capsys.readouterr().out
    assert code == 0
    assert "Completion ingest finished successfully." in out

    def _boom_api_run_ingest(config, **kwargs):
        _ = config, kwargs
        raise RuntimeError("completion failed")

    monkeypatch.setattr(shell_cli, "api_run_ingest", _boom_api_run_ingest)
    code2 = shell_cli.run_db_complete_cli("config.json")
    out2 = capsys.readouterr().out
    assert code2 == 1
    assert "Error: completion failed" in out2


def test_run_migrate_cli_success_and_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    @dataclass
    class _ConfigStub:
        vectordb: VectorDBConfig
        ingest: object
        global_settings: object

    cfg = _ConfigStub(
        vectordb=VectorDBConfig(provider="sqlite", path=Path(".lsm"), collection="kb"),
        ingest=SimpleNamespace(chunking_strategy="structure", chunk_size=1800, chunk_overlap=200),
        global_settings=SimpleNamespace(embed_model="test-model", embedding_dimension=384),
    )
    monkeypatch.setattr(shell_cli, "_load_config", lambda p: cfg)
    monkeypatch.setattr(
        shell_cli,
        "migrate_db",
        lambda **kwargs: {"migrated_vectors": 2, "total_vectors": 2, "validated_tables": 3},
    )

    ok_code = shell_cli.run_migrate_cli(
        "config.json",
        migration_source="sqlite",
        migration_target="sqlite",
    )
    ok_out = capsys.readouterr().out
    assert ok_code == 0
    assert "Migration complete" in ok_out

    monkeypatch.setattr(
        shell_cli,
        "migrate_db",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("migration failed")),
    )
    bad_code = shell_cli.run_migrate_cli(
        "config.json",
        migration_source="sqlite",
        migration_target="sqlite",
    )
    bad_out = capsys.readouterr().out
    assert bad_code == 1
    assert "Error: migration failed" in bad_out
