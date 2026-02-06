from __future__ import annotations

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


class _Container:
    def __init__(self, children=None) -> None:
        self.children = children or []
        self.mounted = []

    def remove_children(self) -> None:
        self.children = []

    def mount(self, child) -> None:
        self.mounted.append(child)
        self.children.append(child)


class _Row:
    def __init__(self, value: str) -> None:
        self._input = SimpleNamespace(value=value)

    def query_one(self, _cls):
        return self._input


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
        roots=["/docs", "/notes"],
        persist_dir="/data",
        collection="kb",
        embed_model="mini",
        device="cpu",
        batch_size=16,
        chunk_size=800,
        chunk_overlap=120,
        tagging_model="gpt-test",
        tags_per_chunk=3,
        enable_ocr=True,
        enable_ai_tagging=False,
    )
    query = SimpleNamespace(
        mode="grounded",
        k=12,
        retrieve_k=None,
        min_relevance=0.2,
        k_rerank=6,
        rerank_strategy="hybrid",
        local_pool=None,
        max_per_file=2,
        path_contains=["docs"],
        ext_allow=[".md"],
        ext_deny=None,
        no_rerank=False,
    )
    vectordb = SimpleNamespace(
        provider="chromadb",
        collection="kb",
        persist_dir="/data",
        chroma_hnsw_space="cosine",
        connection_string=None,
        host=None,
        port=None,
        database=None,
        user=None,
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
    llm_provider = SimpleNamespace(
        provider_name="openai",
        base_url=None,
        endpoint=None,
        api_version=None,
        deployment_name=None,
        query=SimpleNamespace(model="gpt-q"),
        tagging=SimpleNamespace(model="gpt-t"),
        ranking=SimpleNamespace(model="gpt-r"),
    )
    llm = SimpleNamespace(llms=[llm_provider])
    return SimpleNamespace(
        config_path="config.json",
        ingest=ingest,
        query=query,
        vectordb=vectordb,
        llm=llm,
        modes={"grounded": mode_cfg},
        get_mode_config=lambda name: mode_cfg,
        notes=notes,
    )


def _screen():
    app = SimpleNamespace(config=_config(), current_context="settings")
    return _TestableSettingsScreen(app)


def test_focus_tab_and_activate_actions() -> None:
    screen = _screen()
    tabs = _Tabs()
    screen.widgets["#settings-tabs"] = tabs

    screen._focus_active_tab()
    assert tabs.focused is True

    screen.action_settings_tab_1()
    assert tabs.active == "settings-config"
    screen.action_settings_tab_6()
    assert tabs.active == "settings-llm"


def test_set_helpers_and_formatters() -> None:
    screen = _screen()
    inp = _InputWidget()
    sw = _SwitchWidget()
    sel = _SelectWidget()
    screen.widgets["#settings-a"] = inp
    screen.widgets["#settings-b"] = sw
    screen.widgets["#settings-c"] = sel

    screen._set_input("settings-a", "x")
    screen._set_switch("settings-b", True)
    screen._set_select_options("settings-c", ["one", "two"])
    screen._set_select_value("settings-c", "two")

    assert inp.value == "x"
    assert sw.value is True
    assert sel.options == [("one", "one"), ("two", "two")]
    assert sel.value == "two"
    assert screen._format_list(["a", "b"]) == "a, b"
    assert screen._format_list(None) == ""
    assert screen._format_optional(3) == "3"
    assert screen._format_optional(None) == ""


def test_collect_roots_and_button_actions() -> None:
    screen = _screen()
    roots_container = _Container(children=[_Row("/a"), _Row(" "), _Row("/b")])
    screen.widgets["#settings-ingest-roots-list"] = roots_container

    assert screen._collect_ingest_roots() == ["/a", "/b"]

    called = {}
    screen._collect_ingest_roots = lambda: ["/a"]  # type: ignore[method-assign]
    screen._render_ingest_roots = lambda roots: called.setdefault("roots", roots)  # type: ignore[method-assign]
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-ingest-root-add")))
    assert called["roots"] == ["/a", ""]

    called.clear()
    screen._collect_ingest_roots = lambda: ["/a", "/b"]  # type: ignore[method-assign]
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="settings-ingest-root-remove-1")))
    assert called["roots"] == ["/b"]


def test_refresh_and_mode_updates() -> None:
    screen = _screen()
    seen = {"inputs": {}, "switches": {}, "selects": {}, "built": 0, "mode": None}
    screen._set_input = lambda fid, val: seen["inputs"].__setitem__(fid, val)  # type: ignore[method-assign]
    screen._set_switch = lambda fid, val: seen["switches"].__setitem__(fid, val)  # type: ignore[method-assign]
    screen._set_select_options = lambda fid, vals: seen["selects"].__setitem__(fid, vals)  # type: ignore[method-assign]
    screen._set_select_value = lambda fid, val: seen["selects"].__setitem__(f"{fid}:value", val)  # type: ignore[method-assign]
    screen._render_ingest_roots = lambda roots: seen["inputs"].__setitem__("roots", roots)  # type: ignore[method-assign]
    screen._build_llm_sections = lambda providers: seen.__setitem__("built", len(providers))  # type: ignore[method-assign]
    original_update = screen._update_mode_settings
    screen._update_mode_settings = lambda name: (seen.__setitem__("mode", name), original_update(name))[1]  # type: ignore[method-assign]

    screen._refresh_settings()
    assert seen["inputs"]["settings-ingest-collection"] == "kb"
    assert seen["selects"]["settings-query-mode"] == ["grounded"]
    assert seen["selects"]["settings-query-mode:value"] == "grounded"
    assert seen["built"] == 1
    assert seen["mode"] == "grounded"


def test_update_mode_settings_and_select_changed() -> None:
    screen = _screen()
    seen = {"inputs": {}, "switches": {}}
    screen._set_input = lambda fid, val: seen["inputs"].__setitem__(fid, val)  # type: ignore[method-assign]
    screen._set_switch = lambda fid, val: seen["switches"].__setitem__(fid, val)  # type: ignore[method-assign]

    screen._update_mode_settings("grounded")
    assert seen["inputs"]["settings-mode-synthesis-style"] == "grounded"
    assert seen["switches"]["settings-mode-local-enabled"] is True
    assert seen["inputs"]["settings-notes-dir"] == "notes"

    called = {}
    screen._update_mode_settings = lambda name: called.setdefault("mode", name)  # type: ignore[method-assign]
    screen.on_select_changed(SimpleNamespace(select=SimpleNamespace(id="settings-query-mode"), value="grounded"))
    assert called["mode"] == "grounded"
