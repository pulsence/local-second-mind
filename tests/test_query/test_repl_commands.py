"""
Tests for query REPL command handling.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lsm.config.models import LSMConfig, IngestConfig, QueryConfig, LLMConfig, VectorDBConfig
from lsm.query.repl import handle_command
from lsm.query.session import SessionState


def _build_config(tmp_path: Path) -> LSMConfig:
    ingest = IngestConfig(
        roots=[tmp_path],
        manifest=tmp_path / "manifest.json",
    )
    query = QueryConfig()
    llm = LLMConfig(provider="openai", model="gpt-5.2")
    vectordb = VectorDBConfig(persist_dir=tmp_path / ".chroma")
    config = LSMConfig(
        ingest=ingest,
        query=query,
        llm=llm,
        vectordb=vectordb,
        config_path=tmp_path / "config.json",
    )
    return config


def test_unknown_command_shows_help(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.model)
    client = Mock()

    called = {"help": False}

    def _help():
        called["help"] = True

    monkeypatch.setattr("lsm.query.repl.print_help", _help)

    handled = handle_command("/doesnotexist", state, client, config, Mock(), Mock())
    assert handled is True
    assert called["help"] is True


def test_mode_set_model_knowledge(tmp_path):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.model)
    client = Mock()

    handled = handle_command("/mode set model_knowledge on", state, client, config, Mock(), Mock())
    assert handled is True
    assert config.get_mode_config().source_policy.model_knowledge.enabled is True


def test_note_custom_filename(tmp_path, monkeypatch):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.model)
    state.last_question = "Test question"
    state.last_answer = "Test answer"
    state.last_local_sources_for_notes = []
    state.last_remote_sources = []

    client = Mock()

    monkeypatch.setattr("lsm.query.repl.edit_note_in_editor", lambda content: content)

    handled = handle_command("/note custom-note", state, client, config, Mock(), Mock())
    assert handled is True

    note_path = tmp_path / "notes" / "custom-note.md"
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "Test question" in content
    assert "Test answer" in content


def test_provider_status_command(tmp_path, monkeypatch, capsys):
    config = _build_config(tmp_path)
    state = SessionState(model=config.llm.model)
    client = Mock()

    monkeypatch.setattr("lsm.query.repl.list_available_providers", lambda: ["openai"])

    provider = Mock()
    provider.health_check.return_value = {
        "status": "available",
        "stats": {"success_count": 1, "failure_count": 0},
    }
    monkeypatch.setattr("lsm.query.repl.create_provider", lambda _: provider)

    handled = handle_command("/provider-status", state, client, config, Mock(), Mock())
    assert handled is True
    captured = capsys.readouterr()
    assert "PROVIDER HEALTH STATUS" in captured.out
