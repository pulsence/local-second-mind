from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def _config():
    ingest = SimpleNamespace(
        roots=[SimpleNamespace(path=Path("/docs"), tags=None, content_type=None)],
        persist_dir=Path("/data"),
        collection="kb",
        chunking_strategy="structure",
        chunk_size=800,
        chunk_overlap=120,
        tags_per_chunk=3,
        translation_target="en",
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
        k_rerank=6,
        min_relevance=0.2,
        rerank_strategy="hybrid",
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
        providers=[],
        services={},
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
        remote_providers=[],
        remote_provider_chains=[],
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


def _screen():
    app = SimpleNamespace(config=_config(), current_context="settings")
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
    assert "Updated" in status.last


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
    assert "Updated" in status.last


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
