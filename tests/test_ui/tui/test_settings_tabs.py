from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Static as TextualStatic

from lsm.config.models import RemoteProviderRef
from lsm.ui.tui.widgets.settings_chats_notes import ChatsNotesSettingsTab
from lsm.ui.tui.widgets.settings_global import GlobalSettingsTab
from lsm.ui.tui.widgets.settings_ingest import IngestSettingsTab
from lsm.ui.tui.widgets.settings_llm import LLMSettingsTab
from lsm.ui.tui.widgets.settings_modes import ModesSettingsTab
from lsm.ui.tui.widgets.settings_query import QuerySettingsTab
from lsm.ui.tui.widgets.settings_remote import RemoteSettingsTab
from lsm.ui.tui.widgets.settings_vectordb import VectorDBSettingsTab


class _InputWidget:
    def __init__(self) -> None:
        self.value = ""
        self.parent = SimpleNamespace(display=True)


class _SwitchWidget:
    def __init__(self) -> None:
        self.value = False
        self.parent = SimpleNamespace(display=True)


class _SelectWidget:
    def __init__(self) -> None:
        self.value = ""
        self.options: list[tuple[str, str]] = []
        self.parent = SimpleNamespace(display=True)

    def set_options(self, options: list[tuple[str, str]]) -> None:
        self.options = options


class _ContainerWidget:
    def __init__(self) -> None:
        self.children: list[Any] = []

    def remove_children(self) -> None:
        self.children = []

    def mount(self, widget: Any) -> None:
        self.children.append(widget)


class _StatusSink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bool]] = []

    def _set_status(self, message: str, error: bool) -> None:
        self.messages.append((message, error))


def _bind_widgets(tab, widgets: dict[str, Any]) -> None:
    def _query_one(selector, _cls=None):
        key = selector if isinstance(selector, str) else selector
        if key in widgets:
            return widgets[key]
        raise KeyError(key)

    tab.query_one = _query_one  # type: ignore[assignment]


def _attach_status_sink(tab, sink: _StatusSink) -> None:
    parent = SimpleNamespace(parent=sink)
    tab._parent = parent  # type: ignore[attr-defined]


def _config() -> Any:
    ingest = SimpleNamespace(
        roots=[SimpleNamespace(path=Path("/docs"), tags=None, content_type=None)],
        chunking_strategy="structure",
        chunk_size=800,
        chunk_overlap=120,
        tags_per_chunk=3,
        translation_target="en",
        extensions=[".md", ".txt"],
        override_extensions=False,
        exclude_dirs=[".git"],
        override_excludes=False,
        dry_run=False,
        skip_errors=True,
        max_files=None,
        max_seconds=None,
        enable_ocr=True,
        enable_ai_tagging=False,
        enable_language_detection=False,
        enable_translation=False,
    )
    query = SimpleNamespace(
        mode="grounded",
        k=12,
        retrieve_k=None,
        k_rerank=6,
        min_relevance=0.2,
        local_pool=None,
        max_per_file=2,
        path_contains=None,
        ext_allow=None,
        ext_deny=None,
        rerank_strategy="hybrid",
        no_rerank=False,
        chat_mode="single",
        enable_query_cache=False,
        query_cache_ttl=3600,
        query_cache_size=100,
        enable_llm_server_cache=True,
    )
    vectordb = SimpleNamespace(
        provider="sqlite",
        collection="kb",
        path=Path("/data"),
        connection_string=None,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
        index_type="hnsw",
        pool_size=5,
    )
    mode_cfg = SimpleNamespace(
        synthesis_style="grounded",
        source_policy=SimpleNamespace(
            local=SimpleNamespace(enabled=True, min_relevance=0.2, k=12, k_rerank=6),
            remote=SimpleNamespace(enabled=False, rank_strategy="weighted", max_results=5, remote_providers=["brave"]),
            model_knowledge=SimpleNamespace(enabled=False, require_label=True),
        ),
    )
    notes = SimpleNamespace(
        enabled=True,
        dir="notes",
        template="default",
        filename_format="timestamp",
        integration="none",
        wikilinks=False,
        backlinks=False,
        include_tags=False,
    )
    chats = SimpleNamespace(enabled=True, dir="Chats", auto_save=True, format="markdown")
    llm = SimpleNamespace(
        providers=[
            SimpleNamespace(
                provider_name="openai",
                api_key=None,
                base_url=None,
                endpoint=None,
                api_version=None,
                deployment_name=None,
            )
        ],
        services={
            "default": SimpleNamespace(
                provider="openai",
                model="gpt-5.2",
                temperature=None,
                max_tokens=None,
            )
        },
    )
    remote_provider = SimpleNamespace(
        name="brave",
        type="web_search",
        weight=1.0,
        api_key=None,
        endpoint=None,
        max_results=5,
        language=None,
        user_agent=None,
        timeout=None,
        min_interval_seconds=None,
        section_limit=None,
        snippet_max_chars=None,
        include_disambiguation=None,
        cache_results=False,
        cache_ttl=86400,
        extra={},
    )
    remote_chain = SimpleNamespace(
        name="research",
        agent_description="",
        links=[SimpleNamespace(source="brave", map=None)],
    )

    cfg = SimpleNamespace(
        config_path=Path("config.json"),
        ingest=ingest,
        query=query,
        vectordb=vectordb,
        llm=llm,
        modes={"grounded": mode_cfg},
        notes=notes,
        chats=chats,
        remote_providers=[remote_provider],
        remote_provider_chains=[remote_chain],
        global_settings=SimpleNamespace(
            global_folder=Path("/global"),
            embed_model="mini",
            device="cpu",
            batch_size=16,
            embedding_dimension=384,
        ),
    )

    cfg.get_mode_config = lambda name: cfg.modes[name]
    return cfg


def test_global_tab_refresh_and_updates() -> None:
    cfg = _config()
    tab = GlobalSettingsTab()
    widgets = {
        "#settings-config-path": _InputWidget(),
        "#settings-global-folder": _InputWidget(),
        "#settings-global-embed-model": _InputWidget(),
        "#settings-global-device": _InputWidget(),
        "#settings-global-batch-size": _InputWidget(),
        "#settings-global-embedding-dimension": _InputWidget(),
    }
    _bind_widgets(tab, widgets)

    tab.refresh_fields(cfg)
    assert widgets["#settings-global-device"].value == "cpu"
    assert widgets["#settings-global-batch-size"].value == "16"

    assert tab.apply_update("settings-global-device", "cuda:0", cfg) is True
    assert tab.apply_update("settings-global-batch-size", "64", cfg) is True
    assert tab.apply_update("settings-global-embedding-dimension", "", cfg) is True
    assert cfg.global_settings.device == "cuda:0"
    assert cfg.global_settings.batch_size == 64
    assert cfg.global_settings.embedding_dimension is None


def test_ingest_tab_dynamic_roots_and_buttons() -> None:
    cfg = _config()
    tab = IngestSettingsTab()
    sink = _StatusSink()
    _attach_status_sink(tab, sink)
    widgets = {
        "#settings-ingest-roots-list": _ContainerWidget(),
        "#settings-ingest-persist-dir": _InputWidget(),
        "#settings-ingest-collection": _InputWidget(),
        "#settings-ingest-manifest": _InputWidget(),
        "#settings-ingest-chroma-flush-interval": _InputWidget(),
        "#settings-ingest-chunking-strategy": _SelectWidget(),
        "#settings-ingest-chunk-size": _InputWidget(),
        "#settings-ingest-chunk-overlap": _InputWidget(),
        "#settings-ingest-tags-per-chunk": _InputWidget(),
        "#settings-ingest-translation-target": _InputWidget(),
        "#settings-ingest-extensions": _InputWidget(),
        "#settings-ingest-exclude-dirs": _InputWidget(),
        "#settings-ingest-max-files": _InputWidget(),
        "#settings-ingest-max-seconds": _InputWidget(),
        "#settings-ingest-override-extensions": _SwitchWidget(),
        "#settings-ingest-override-excludes": _SwitchWidget(),
        "#settings-ingest-dry-run": _SwitchWidget(),
        "#settings-ingest-skip-errors": _SwitchWidget(),
        "#settings-ingest-enable-ocr": _SwitchWidget(),
        "#settings-ingest-enable-ai-tagging": _SwitchWidget(),
        "#settings-ingest-enable-language-detection": _SwitchWidget(),
        "#settings-ingest-enable-translation": _SwitchWidget(),
        "#settings-ingest-enable-versioning": _SwitchWidget(),
    }
    _bind_widgets(tab, widgets)
    tab._field = lambda *args, **kwargs: TextualStatic("")  # type: ignore[assignment]

    tab.refresh_fields(cfg)
    assert len(widgets["#settings-ingest-roots-list"].children) >= 1

    assert tab.apply_update("settings-ingest-root-0-path", "/docs/new", cfg) is True
    assert tab.apply_update("settings-ingest-root-0-tags", "a, b", cfg) is True
    assert cfg.ingest.roots[0].path == Path("/docs/new")
    assert cfg.ingest.roots[0].tags == ["a", "b"]

    assert tab.handle_button("settings-ingest-root-add", cfg) is True
    assert len(cfg.ingest.roots) == 2
    assert tab.handle_button("settings-ingest-root-remove-1", cfg) is True
    assert len(cfg.ingest.roots) == 1
    assert sink.messages[-1][0] == "Removed ingest root"


def test_query_tab_refresh_and_no_rerank_update() -> None:
    cfg = _config()
    tab = QuerySettingsTab()
    widgets = {
        "#settings-query-mode": _SelectWidget(),
        "#settings-query-k": _InputWidget(),
        "#settings-query-retrieve-k": _InputWidget(),
        "#settings-query-k-rerank": _InputWidget(),
        "#settings-query-min-relevance": _InputWidget(),
        "#settings-query-local-pool": _InputWidget(),
        "#settings-query-max-per-file": _InputWidget(),
        "#settings-query-path-contains": _InputWidget(),
        "#settings-query-ext-allow": _InputWidget(),
        "#settings-query-ext-deny": _InputWidget(),
        "#settings-query-rerank-strategy": _SelectWidget(),
        "#settings-query-chat-mode": _SelectWidget(),
        "#settings-query-cache-ttl": _InputWidget(),
        "#settings-query-cache-size": _InputWidget(),
        "#settings-query-no-rerank": _SwitchWidget(),
        "#settings-query-enable-cache": _SwitchWidget(),
        "#settings-query-enable-llm-server-cache": _SwitchWidget(),
    }
    _bind_widgets(tab, widgets)

    tab.refresh_fields(cfg)
    assert widgets["#settings-query-mode"].options == [("grounded", "grounded")]
    assert widgets["#settings-query-k"].value == "12"

    assert tab.apply_update("settings-query-no-rerank", True, cfg) is True
    assert cfg.query.no_rerank is True
    assert cfg.query.rerank_strategy == "none"


def test_llm_tab_refresh_update_and_buttons() -> None:
    cfg = _config()
    tab = LLMSettingsTab()
    sink = _StatusSink()
    _attach_status_sink(tab, sink)
    widgets = {
        "#settings-llm-services-list": _ContainerWidget(),
        "#settings-llm-providers-list": _ContainerWidget(),
    }
    _bind_widgets(tab, widgets)
    tab._field = lambda *args, **kwargs: TextualStatic("")  # type: ignore[assignment]

    tab.refresh_fields(cfg)
    assert len(widgets["#settings-llm-services-list"].children) >= 1
    assert len(widgets["#settings-llm-providers-list"].children) >= 1

    assert tab.apply_update("settings-llm-provider-0-name", "openai-main", cfg) is True
    assert cfg.llm.providers[0].provider_name == "openai-main"
    assert cfg.llm.services["default"].provider == "openai-main"

    assert tab.handle_button("settings-llm-provider-add", cfg) is True
    assert tab.handle_button("settings-llm-service-add", cfg) is True
    assert len(cfg.llm.providers) == 2
    assert len(cfg.llm.services) == 2

    assert tab.handle_button("settings-llm-service-remove-1", cfg) is True
    assert len(cfg.llm.services) == 1


def test_vectordb_tab_refresh_updates_and_visibility() -> None:
    cfg = _config()
    tab = VectorDBSettingsTab()
    widgets = {
        "#settings-vdb-provider": _SelectWidget(),
        "#settings-vdb-collection": _InputWidget(),
        "#settings-vdb-path": _InputWidget(),
        "#settings-vdb-connection-string": _InputWidget(),
        "#settings-vdb-host": _InputWidget(),
        "#settings-vdb-port": _InputWidget(),
        "#settings-vdb-database": _InputWidget(),
        "#settings-vdb-user": _InputWidget(),
        "#settings-vdb-password": _InputWidget(),
        "#settings-vdb-index-type": _InputWidget(),
        "#settings-vdb-pool-size": _InputWidget(),
    }
    _bind_widgets(tab, widgets)

    tab.refresh_fields(cfg)
    assert widgets["#settings-vdb-provider"].value == "sqlite"
    assert widgets["#settings-vdb-connection-string"].parent.display is False

    assert tab.apply_update("settings-vdb-provider", "postgresql", cfg) is True
    assert cfg.vectordb.provider == "postgresql"
    assert widgets["#settings-vdb-connection-string"].parent.display is True
    assert widgets["#settings-vdb-path"].parent.display is False


def test_modes_tab_refresh_and_mode_change() -> None:
    cfg = _config()
    tab = ModesSettingsTab()
    widgets = {
        "#settings-modes-mode": _SelectWidget(),
        "#settings-modes-synthesis-style": _InputWidget(),
        "#settings-modes-local-policy": _InputWidget(),
        "#settings-modes-remote-policy": _InputWidget(),
        "#settings-modes-model-policy": _InputWidget(),
    }
    _bind_widgets(tab, widgets)

    cfg.modes["hybrid"] = cfg.modes["grounded"]

    tab.refresh_fields(cfg)
    assert widgets["#settings-modes-mode"].value == "grounded"
    assert "grounded" in widgets["#settings-modes-synthesis-style"].value

    assert tab.apply_update("settings-modes-mode", "hybrid", cfg) is True
    assert cfg.query.mode == "hybrid"


def test_remote_tab_updates_and_buttons() -> None:
    cfg = _config()
    cfg.modes["grounded"].source_policy.remote.remote_providers = [
        "brave",
        RemoteProviderRef(source="brave", weight=0.6),
    ]

    tab = RemoteSettingsTab()
    sink = _StatusSink()
    _attach_status_sink(tab, sink)
    widgets = {
        "#settings-remote-providers-list": _ContainerWidget(),
        "#settings-remote-chains-list": _ContainerWidget(),
    }
    _bind_widgets(tab, widgets)
    tab._field = lambda *args, **kwargs: TextualStatic("")  # type: ignore[assignment]

    tab.refresh_fields(cfg)
    assert len(widgets["#settings-remote-providers-list"].children) >= 1
    assert len(widgets["#settings-remote-chains-list"].children) >= 1

    assert tab.apply_update("settings-remote-provider-0-name", "brave-main", cfg) is True
    assert cfg.remote_providers[0].name == "brave-main"
    assert cfg.remote_provider_chains[0].links[0].source == "brave-main"
    refs = cfg.modes["grounded"].source_policy.remote.remote_providers
    assert refs[0] == "brave-main"
    assert isinstance(refs[1], RemoteProviderRef)
    assert refs[1].source == "brave-main"

    assert tab.apply_update("settings-remote-chain-0-link-0-map", "doi:doi", cfg) is True
    assert cfg.remote_provider_chains[0].links[0].map == ["doi:doi"]

    assert tab.handle_button("settings-remote-provider-add", cfg) is True
    assert tab.handle_button("settings-remote-chain-add", cfg) is True
    assert tab.handle_button("settings-remote-chain-0-link-add", cfg) is True
    assert len(cfg.remote_providers) == 2
    assert len(cfg.remote_provider_chains) == 2
    assert len(cfg.remote_provider_chains[0].links) == 2

    assert tab.handle_button("settings-remote-chain-0-link-remove-1", cfg) is True
    assert len(cfg.remote_provider_chains[0].links) == 1


@pytest.mark.parametrize(
    "field_id,value,expected",
    [
        ("settings-chats-enabled", False, False),
        ("settings-chats-dir", "Chats2", "Chats2"),
        ("settings-notes-integration", "obsidian", "obsidian"),
        ("settings-notes-wikilinks", True, True),
    ],
)
def test_chats_notes_tab_refresh_and_updates(field_id: str, value: Any, expected: Any) -> None:
    cfg = _config()
    tab = ChatsNotesSettingsTab()
    widgets = {
        "#settings-chats-enabled": _SwitchWidget(),
        "#settings-chats-dir": _InputWidget(),
        "#settings-chats-auto-save": _SwitchWidget(),
        "#settings-chats-format": _InputWidget(),
        "#settings-notes-enabled": _SwitchWidget(),
        "#settings-notes-dir": _InputWidget(),
        "#settings-notes-template": _InputWidget(),
        "#settings-notes-filename-format": _InputWidget(),
        "#settings-notes-integration": _InputWidget(),
        "#settings-notes-wikilinks": _SwitchWidget(),
        "#settings-notes-backlinks": _SwitchWidget(),
        "#settings-notes-include-tags": _SwitchWidget(),
    }
    _bind_widgets(tab, widgets)

    tab.refresh_fields(cfg)
    assert widgets["#settings-chats-dir"].value == "Chats"

    assert tab.apply_update(field_id, value, cfg) is True
    if field_id.startswith("settings-chats-"):
        attr = field_id.removeprefix("settings-chats-").replace("-", "_")
        assert getattr(cfg.chats, attr) == expected
    else:
        attr = field_id.removeprefix("settings-notes-").replace("-", "_")
        assert getattr(cfg.notes, attr) == expected
