from __future__ import annotations

from pathlib import Path

from lsm.config.loader import build_config_from_raw, config_to_raw
from lsm.ui.tui.state import SettingsViewModel


def _raw_config(tmp_path: Path) -> dict:
    return {
        "global": {
            "global_folder": str(tmp_path / "lsm-global"),
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 16,
        },
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "chunk_size": 800,
            "chunk_overlap": 100,
        },
        "llms": {
            "providers": [
                {
                    "provider_name": "openai",
                    "api_key": "test-key",
                }
            ],
            "services": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-5.2",
                }
            },
        },
        "vectordb": {
            "provider": "sqlite",
            "path": str(tmp_path / "data"),
            "collection": "test_collection",
        },
        "query": {
            "mode": "grounded",
            "k": 12,
            "k_rerank": 6,
            "min_relevance": 0.25,
        },
        "modes": [
            {
                "name": "grounded",
                "synthesis_style": "grounded",
                "source_policy": {
                    "local": {
                        "min_relevance": 0.25,
                        "k": 12,
                        "k_rerank": 6,
                    },
                    "remote": {
                        "enabled": True,
                        "remote_providers": ["brave"],
                    },
                    "model_knowledge": {
                        "enabled": False,
                    },
                },
            },
            {
                "name": "insight",
                "synthesis_style": "insight",
                "source_policy": {
                    "local": {
                        "min_relevance": 0.2,
                        "k": 10,
                        "k_rerank": 4,
                    },
                    "remote": {
                        "enabled": False,
                    },
                    "model_knowledge": {
                        "enabled": True,
                    },
                },
            },
        ],
        "remote_providers": [
            {
                "name": "brave",
                "type": "web_search",
                "weight": 1.0,
            }
        ],
        "remote_provider_chains": [
            {
                "name": "research",
                "links": [
                    {
                        "source": "brave",
                    }
                ],
            }
        ],
    }


def _config(tmp_path: Path):
    raw = _raw_config(tmp_path)
    return build_config_from_raw(raw, tmp_path / "config.json")


def test_update_field_tracks_dirty_and_keeps_persisted(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    result = vm.update_field("settings-query-k", "20")

    assert result.handled is True
    assert "settings-query" in result.changed_tabs
    assert vm.draft_config.query.k == 20
    assert vm.persisted_config.query.k == 12
    assert "settings-query-k" in vm.dirty_fields
    assert "settings-query" in vm.dirty_tabs


def test_mode_update_marks_query_and_modes_tabs(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    result = vm.update_field("settings-query-mode", "insight")

    assert result.handled is True
    assert set(result.changed_tabs) == {"settings-query", "settings-modes"}
    assert vm.draft_config.query.mode == "insight"
    assert "settings-query" in vm.dirty_tabs
    assert "settings-modes" in vm.dirty_tabs


def test_add_and_remove_items_updates_draft(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    add_result = vm.add_item("settings-llm-provider-add")

    assert add_result.handled is True
    assert len(vm.draft_config.llm.providers) == 2

    remove_result = vm.remove_item("settings-llm-provider-remove-1")

    assert remove_result.handled is True
    assert len(vm.draft_config.llm.providers) == 1


def test_remote_provider_rename_updates_refs_and_chains(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    result = vm.update_field("settings-remote-provider-0-name", "brave-main")

    assert result.handled is True
    assert vm.draft_config.remote_providers is not None
    assert vm.draft_config.remote_providers[0].name == "brave-main"
    assert vm.draft_config.remote_provider_chains is not None
    assert vm.draft_config.remote_provider_chains[0].links[0].source == "brave-main"
    refs = vm.draft_config.modes["grounded"].source_policy.remote.remote_providers
    assert refs is not None
    assert refs[0] == "brave-main"


def test_reset_tab_reverts_only_target_section(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    vm.update_field("settings-query-k", "42")
    vm.update_field("settings-global-device", "cuda:0")

    result = vm.reset_tab("settings-query")

    assert result.handled is True
    assert vm.draft_config.query.k == vm.persisted_config.query.k
    assert vm.draft_config.global_settings.device == "cuda:0"
    assert "settings-query" not in vm.dirty_tabs
    assert "settings-global" in vm.dirty_tabs


def test_save_updates_persisted_state_and_clears_dirty(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    vm.update_field("settings-query-k", "21")

    calls: list[tuple[object, object]] = []

    def _saver(config, path) -> None:
        calls.append((config, path))

    result = vm.save(saver=_saver)

    assert result.handled is True
    assert result.error is None
    assert len(calls) == 1
    assert vm.persisted_config.query.k == 21
    assert vm.dirty_tabs == frozenset()
    assert vm.dirty_fields == frozenset()


def test_save_validation_error_is_reported(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    vm.update_field("settings-ingest-enable-translation", True)

    result = vm.save(saver=lambda _config, _path: None)

    assert result.handled is True
    assert result.error is not None
    assert "enable_language_detection" in result.error
    assert "_save" in vm.validation_errors


def test_reset_all_restores_persisted_snapshot(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    vm.update_field("settings-query-k", "99")
    vm.update_field("settings-global-device", "cuda:0")

    vm.reset_all()

    assert config_to_raw(vm.draft_config) == config_to_raw(vm.persisted_config)
    assert vm.dirty_tabs == frozenset()
    assert vm.dirty_fields == frozenset()


def test_set_key_updates_global_path(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    new_folder = tmp_path / "new-global"

    result = vm.set_key("settings-global", "global_folder", str(new_folder))

    assert result.handled is True
    assert result.error is None
    assert vm.draft_config.global_settings.global_folder == new_folder.resolve()
    assert "settings-global" in vm.dirty_tabs


def test_unset_key_sets_optional_value_to_none(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    vm.set_key("settings-vdb", "connection_string", "postgresql://user:pass@localhost/db")

    result = vm.unset_key("settings-vdb", "connection_string")

    assert result.handled is True
    assert result.error is None
    assert vm.draft_config.vectordb.connection_string is None


def test_reset_key_restores_persisted_scalar(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    vm.set_key("settings-query", "k", "99")

    result = vm.reset_key("settings-query", "k")

    assert result.handled is True
    assert result.error is None
    assert vm.draft_config.query.k == vm.persisted_config.query.k == 12


def test_default_key_deletes_path_and_uses_model_default(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    vm.set_key("settings-query", "path_contains", '["docs", "notes"]')
    assert vm.draft_config.query.path_contains == ["docs", "notes"]

    result = vm.default_key("settings-query", "path_contains")

    assert result.handled is True
    assert result.error is None
    assert vm.draft_config.query.path_contains is None


def test_table_rows_marks_dirty_values(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))
    vm.set_key("settings-query", "k", "77")

    rows = vm.table_rows("settings-query")
    row = next(item for item in rows if item.key == "k")

    assert row.value == "77"
    assert row.state == "dirty"


def test_llm_tab_rows_render_from_llms_section(tmp_path: Path) -> None:
    vm = SettingsViewModel(_config(tmp_path))

    rows = vm.table_rows("settings-llm")
    keys = {row.key for row in rows}

    assert "providers[0].provider_name" in keys
    assert "services.default.model" in keys
