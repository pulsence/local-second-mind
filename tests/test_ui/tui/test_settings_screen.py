from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lsm.config.models import RemoteProviderRef
from lsm.ui.tui.screens.settings import SettingsScreen


class _Tabs:
    def __init__(self) -> None:
        self.active = ""
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _InputWidget:
    def __init__(self) -> None:
        self.value = ""


class _SwitchWidget:
    def __init__(self) -> None:
        self.value = False


class _SelectWidget:
    def __init__(self) -> None:
        self.value = ""
        self.options = []

    def set_options(self, options):
        self.options = options


class _StaticWidget:
    def __init__(self) -> None:
        self.last = ""

    def update(self, value: str) -> None:
        self.last = value


class _ContainerWidget:
    def __init__(self) -> None:
        self.children = []

    def remove_children(self) -> None:
        self.children = []

    def mount(self, widget) -> None:
        self.children.append(widget)


class _TestableSettingsScreen(SettingsScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self.widgets = {}

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        key = selector if isinstance(selector, str) else selector
        if key in self.widgets:
            return self.widgets[key]
        raise KeyError(key)

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()


class _DeferredRefreshSettingsScreen(_TestableSettingsScreen):
    def __init__(self, app):
        super().__init__(app)
        self.after_refresh = None

    def call_after_refresh(self, fn):  # type: ignore[override]
        self.after_refresh = fn


def _config():
    ingest = SimpleNamespace(
        roots=[SimpleNamespace(path=Path("/docs"), tags=None, content_type=None)],
        persist_dir=Path("/data"),
        collection="kb",
        manifest=Path("/data/manifest.json"),
        chroma_flush_interval=500,
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
        enable_versioning=False,
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
        provider="chromadb",
        collection="kb",
        persist_dir=Path("/data"),
        chroma_hnsw_space="cosine",
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
            remote=SimpleNamespace(enabled=False, rank_strategy="weighted", max_results=5, remote_providers=["wiki"]),
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
    return SimpleNamespace(
        config_path=Path("config.json"),
        ingest=ingest,
        query=query,
        vectordb=vectordb,
        llm=llm,
        modes={"grounded": mode_cfg},
        get_mode_config=lambda name: mode_cfg,
        notes=notes,
        chats=chats,
        remote_providers=[remote_provider],
        remote_provider_chains=[remote_chain],
        global_folder=Path("/global"),
        embed_model="mini",
        device="cpu",
        batch_size=16,
        embedding_dimension=384,
        global_settings=SimpleNamespace(
            global_folder=Path("/global"),
            embed_model="mini",
            device="cpu",
            batch_size=16,
            embedding_dimension=384,
        ),
    )


def _screen(context: str = "settings"):
    app = SimpleNamespace(config=_config(), current_context=context)
    return _TestableSettingsScreen(app)


def test_focus_and_tab_actions() -> None:
    screen = _screen()
    tabs = _Tabs()
    screen.widgets["#settings-tabs"] = tabs

    screen.on_mount()
    assert tabs.focused is True

    screen.action_settings_tab_1()
    assert tabs.active == "settings-global"
    screen.action_settings_tab_8()
    assert tabs.active == "settings-chats-notes"


def test_live_update_query_fields() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    cfg = screen.app.config
    screen._apply_live_update("settings-query-k", "20")
    screen._apply_live_update("settings-query-min-relevance", "0.5")
    screen._apply_live_update("settings-query-chat-mode", "chat")
    screen._apply_live_update("settings-global-device", "cuda:0")

    assert cfg.query.k == 20
    assert cfg.query.min_relevance == 0.5
    assert cfg.query.chat_mode == "chat"
    assert cfg.global_settings.device == "cuda:0"
    assert status.last == ""


def test_live_switch_updates() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    screen._apply_live_switch_update("settings-query-enable-cache", True)
    screen._apply_live_switch_update("settings-ingest-enable-translation", True)
    screen._apply_live_switch_update("settings-notes-enabled", False)

    assert cfg.query.enable_query_cache is True
    assert cfg.ingest.enable_translation is True
    assert cfg.notes.enabled is False
    assert status.last == ""


def test_save_and_reset_buttons(monkeypatch) -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    saved = {"called": False}
    reset = {"called": False}
    monkeypatch.setattr(screen, "_save_config", lambda: saved.__setitem__("called", True))
    monkeypatch.setattr(screen, "_reset_config", lambda: reset.__setitem__("called", True))

    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-save-global")))
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-reset-global")))

    assert saved["called"] is True
    assert reset["called"] is True


def test_mode_display_updates_fields() -> None:
    screen = _screen()
    mode_name = _InputWidget()
    local_policy = _InputWidget()
    remote_policy = _InputWidget()
    model_policy = _InputWidget()
    screen.widgets["#settings-modes-synthesis-style"] = mode_name
    screen.widgets["#settings-modes-local-policy"] = local_policy
    screen.widgets["#settings-modes-remote-policy"] = remote_policy
    screen.widgets["#settings-modes-model-policy"] = model_policy

    screen._update_mode_display("grounded")
    assert "grounded" in mode_name.value
    assert "enabled=True" in local_policy.value
    assert "rank=weighted" in remote_policy.value
    assert "require_label=True" in model_policy.value


def test_refresh_loads_enable_ocr_switch_value() -> None:
    screen = _screen()
    ocr_switch = _SwitchWidget()
    screen.widgets["#settings-ingest-enable-ocr"] = ocr_switch

    screen._refresh_settings()
    assert ocr_switch.value is True


def test_refresh_sets_additional_ingest_and_query_fields() -> None:
    screen = _screen()
    manifest = _InputWidget()
    retrieve_k = _InputWidget()
    no_rerank = _SwitchWidget()
    screen.widgets["#settings-ingest-manifest"] = manifest
    screen.widgets["#settings-query-retrieve-k"] = retrieve_k
    screen.widgets["#settings-query-no-rerank"] = no_rerank

    screen._refresh_settings()

    assert Path(manifest.value) == Path("/data/manifest.json")
    assert retrieve_k.value == ""
    assert no_rerank.value is False


def test_on_show_refreshes_values_even_if_context_is_stale() -> None:
    screen = _screen(context="query")
    ocr_switch = _SwitchWidget()
    screen.widgets["#settings-ingest-enable-ocr"] = ocr_switch

    screen.on_show()
    assert ocr_switch.value is True


def test_refresh_keeps_guard_until_after_refresh_callback_runs() -> None:
    app = SimpleNamespace(config=_config(), current_context="settings")
    screen = _DeferredRefreshSettingsScreen(app)

    screen._refresh_settings()

    assert screen._is_refreshing is True
    assert callable(screen.after_refresh)

    screen.after_refresh()
    assert screen._is_refreshing is False


def test_settings_tab_activation_does_not_refresh_values() -> None:
    screen = _screen(context="query")
    ocr_switch = _SwitchWidget()
    screen.widgets["#settings-ingest-enable-ocr"] = ocr_switch

    event = SimpleNamespace(tabbed_content=SimpleNamespace(id="settings-tabs"))
    screen.on_tabbed_content_tab_activated(event)
    assert ocr_switch.value is False


def test_dynamic_llm_updates() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    handled_provider = screen._apply_dynamic_text_update("settings-llm-provider-0-name", "openai-main")
    handled_service = screen._apply_dynamic_text_update("settings-llm-service-0-model", "gpt-5.2-mini")

    assert handled_provider is True
    assert handled_service is True
    assert cfg.llm.providers[0].provider_name == "openai-main"
    assert cfg.llm.services["default"].model == "gpt-5.2-mini"
    assert status.last == ""


def test_dynamic_llm_provider_rename_updates_service_provider() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    handled = screen._apply_dynamic_text_update("settings-llm-provider-0-name", "openai-main")

    assert handled is True
    assert cfg.llm.providers[0].provider_name == "openai-main"
    assert cfg.llm.services["default"].provider == "openai-main"
    assert status.last == ""


def test_dynamic_llm_provider_name_no_refresh_when_unchanged(monkeypatch) -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status

    called = {"count": 0}

    def _refresh(_cfg) -> None:
        called["count"] += 1

    monkeypatch.setattr(screen, "_refresh_llm_structured_fields", _refresh)

    handled = screen._apply_dynamic_text_update("settings-llm-provider-0-name", "openai")
    assert handled is True
    assert called["count"] == 0


def test_dynamic_ingest_root_updates() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    handled_path = screen._apply_dynamic_text_update("settings-ingest-root-0-path", "/docs/new")
    handled_tags = screen._apply_dynamic_text_update("settings-ingest-root-0-tags", "a, b, c")
    handled_content_type = screen._apply_dynamic_text_update(
        "settings-ingest-root-0-content-type",
        "notes",
    )

    assert handled_path is True
    assert handled_tags is True
    assert handled_content_type is True
    assert cfg.ingest.roots[0].path == Path("/docs/new")
    assert cfg.ingest.roots[0].tags == ["a", "b", "c"]
    assert cfg.ingest.roots[0].content_type == "notes"
    assert status.last == ""


def test_dynamic_remote_updates() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    handled_weight = screen._apply_dynamic_text_update("settings-remote-provider-0-weight", "0.75")
    handled_cache = screen._apply_dynamic_switch_update("settings-remote-provider-0-cache-results", True)
    handled_map = screen._apply_dynamic_text_update(
        "settings-remote-chain-0-link-0-map",
        "doi:doi, title:query",
    )

    assert handled_weight is True
    assert handled_cache is True
    assert handled_map is True
    assert cfg.remote_providers[0].weight == 0.75
    assert cfg.remote_providers[0].cache_results is True
    assert cfg.remote_provider_chains[0].links[0].map == ["doi:doi", "title:query"]
    assert status.last == ""


def test_dynamic_remote_provider_rename_updates_refs_and_chain_links() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    cfg.modes["grounded"].source_policy.remote.remote_providers = [
        "brave",
        RemoteProviderRef(source="brave", weight=0.5),
    ]
    cfg.remote_provider_chains[0].links[0].source = "brave"

    handled = screen._apply_dynamic_text_update("settings-remote-provider-0-name", "brave-main")

    assert handled is True
    assert cfg.remote_providers[0].name == "brave-main"
    refs = cfg.modes["grounded"].source_policy.remote.remote_providers
    assert refs[0] == "brave-main"
    assert isinstance(refs[1], RemoteProviderRef)
    assert refs[1].source == "brave-main"
    assert cfg.remote_provider_chains[0].links[0].source == "brave-main"
    assert status.last == ""


def test_refresh_mounts_llm_services_and_remote_chains_lists(monkeypatch) -> None:
    from textual.widgets import Static as TextualStatic

    screen = _screen()
    llm_services = _ContainerWidget()
    remote_providers = _ContainerWidget()
    remote_chains = _ContainerWidget()
    screen.widgets["#settings-llm-services-list"] = llm_services
    screen.widgets["#settings-remote-providers-list"] = remote_providers
    screen.widgets["#settings-remote-chains-list"] = remote_chains
    monkeypatch.setattr(screen, "_field", lambda *args, **kwargs: TextualStatic(""))

    screen._refresh_settings()

    assert len(llm_services.children) >= 1
    assert len(remote_providers.children) >= 1
    assert len(remote_chains.children) >= 1


def test_structured_buttons_add_records() -> None:
    screen = _screen()
    status = _StaticWidget()
    screen.widgets["#settings-status"] = status
    cfg = screen.app.config

    base_llm_providers = len(cfg.llm.providers)
    base_llm_services = len(cfg.llm.services)
    base_roots = len(cfg.ingest.roots)
    base_remote_providers = len(cfg.remote_providers)
    base_remote_chains = len(cfg.remote_provider_chains)

    assert screen._handle_structured_button("settings-ingest-root-add") is True
    assert screen._handle_structured_button("settings-llm-provider-add") is True
    assert screen._handle_structured_button("settings-llm-service-add") is True
    assert screen._handle_structured_button("settings-remote-provider-add") is True
    assert screen._handle_structured_button("settings-remote-chain-add") is True
    assert screen._handle_structured_button("settings-remote-chain-0-link-add") is True

    assert len(cfg.ingest.roots) == base_roots + 1
    assert len(cfg.llm.providers) == base_llm_providers + 1
    assert len(cfg.llm.services) == base_llm_services + 1
    assert len(cfg.remote_providers) == base_remote_providers + 1
    assert len(cfg.remote_provider_chains) == base_remote_chains + 1
    assert len(cfg.remote_provider_chains[0].links) == 2
