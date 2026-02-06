from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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

    def _api_run_ingest(config, force=False, progress_callback=None):
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
