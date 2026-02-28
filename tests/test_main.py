from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import lsm.__main__ as lsm_main


def test_main_missing_config_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)

    code = lsm_main.main(["--config", str(missing)])
    out = capsys.readouterr().out
    assert code == 1
    assert "Configuration file not found" in out


def test_main_runs_tui_when_no_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)
    monkeypatch.setattr("lsm.ui.tui.app.run_tui", lambda cfg: 7)

    code = lsm_main.main(["--config", str(cfg_file)])
    assert code == 7


def test_main_dispatches_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)
    monkeypatch.setattr("lsm.ui.shell.cli.run_ingest", lambda args: 9)

    code = lsm_main.main(["--config", str(cfg_file), "ingest", "build"])
    assert code == 9


def test_main_dispatches_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)
    monkeypatch.setattr("lsm.ui.shell.cli.run_db", lambda args: 5)

    code = lsm_main.main(["--config", str(cfg_file), "db", "prune"])
    assert code == 5


def test_main_dispatches_migrate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)
    monkeypatch.setattr("lsm.ui.shell.cli.run_migrate", lambda args: 6)

    code = lsm_main.main(["--config", str(cfg_file), "migrate", "--from", "sqlite", "--to", "sqlite"])
    assert code == 6


def test_main_handles_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)

    def _raise_interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr("lsm.ui.shell.cli.run_ingest", _raise_interrupt)
    code = lsm_main.main(["--config", str(cfg_file), "ingest", "build"])
    out = capsys.readouterr().out
    assert code == 130
    assert "Interrupted by user" in out


def test_main_handles_exception_verbose_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)

    def _boom(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr("lsm.ui.shell.cli.run_ingest", _boom)
    code = lsm_main.main(["--config", str(cfg_file), "ingest", "build"])
    err = capsys.readouterr().err
    assert code == 1
    assert "Error: boom" in err


def test_main_reraises_when_verbose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}", encoding="utf-8")
    fake_config = SimpleNamespace()

    monkeypatch.setattr(lsm_main, "configure_logging_from_args", lambda **kwargs: None)
    monkeypatch.setattr("lsm.config.load_config_from_file", lambda p: fake_config)
    monkeypatch.setattr("lsm.ui.shell.cli.run_ingest", lambda _args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        lsm_main.main(["-v", "--config", str(cfg_file), "ingest", "build"])
