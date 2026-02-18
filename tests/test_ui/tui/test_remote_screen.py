from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lsm.ui.tui.screens.remote import RemoteScreen


class _Input:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _Select:
    def __init__(self, value: str = "", widget_id: str = "remote-provider-select") -> None:
        self.value = value
        self.id = widget_id
        self.options = []
        self.focused = False

    def set_options(self, options) -> None:
        self.options = options

    def focus(self) -> None:
        self.focused = True


class _Static:
    def __init__(self) -> None:
        self.last = ""

    def update(self, message: str) -> None:
        self.last = message


class _FakeDocument:
    @property
    def end(self):
        return (0, 0)


class _TextArea:
    def __init__(self) -> None:
        self.lines = []
        self.ended = False
        self.document = _FakeDocument()

    def insert(self, text: str, location) -> None:
        self.lines.append(text)

    def scroll_end(self) -> None:
        self.ended = True


class _Provider:
    def __init__(self, name: str, type_: str, weight: float = 1.0) -> None:
        self.name = name
        self.type = type_
        self.weight = weight


class _Config:
    def __init__(self, providers: list[_Provider]) -> None:
        self.remote_providers = providers
        self.query = SimpleNamespace(mode="grounded")
        self.llm = SimpleNamespace(get_query_config=lambda: SimpleNamespace(model="gpt-test"))
        self._weights = {p.name: p.weight for p in providers}

    def get_mode_config(self):
        return SimpleNamespace(
            source_policy=SimpleNamespace(
                remote=SimpleNamespace(
                    enabled=True,
                    max_results=5,
                    rank_strategy="weighted",
                    remote_providers=["wiki"],
                )
            )
        )

    def set_remote_provider_weight(self, name: str, weight: float) -> bool:
        for provider in self.remote_providers:
            if provider.name == name:
                provider.weight = weight
                self._weights[name] = weight
                return True
        return False


class _TestableRemoteScreen(RemoteScreen):
    def __init__(self, app) -> None:
        super().__init__()
        self._test_app = app
        self.widgets = {}
        self.worker_calls = []

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        if selector in self.widgets:
            return self.widgets[selector]
        raise KeyError(selector)

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()

    def run_worker(self, coro, exclusive=False):  # type: ignore[override]
        self.worker_calls.append((coro, exclusive))
        coro.close()


def _screen(context: str = "remote", providers: list[_Provider] | None = None):
    providers = providers if providers is not None else [_Provider("wiki", "wikipedia", 1.2)]
    app = SimpleNamespace(
        config=_Config(providers),
        current_context=context,
        _tui_log_buffer=["line1", "line2"],
        query_state=None,
    )
    screen = _TestableRemoteScreen(app)
    screen.widgets["#remote-provider-list"] = _Static()
    screen.widgets["#remote-provider-select"] = _Select("")
    screen.widgets["#remote-query-input"] = _Input("", "remote-query-input")
    screen.widgets["#remote-weight-input"] = _Input("")
    screen.widgets["#remote-results-output"] = _Static()
    screen.widgets["#remote-log"] = _TextArea()
    return screen


def test_on_mount_and_provider_list(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    monkeypatch.setattr("lsm.ui.tui.screens.remote.get_registered_providers", lambda: {"wikipedia": object(), "brave": object()})

    screen.on_mount()

    provider_list = screen.widgets["#remote-provider-list"].last
    assert "Configured Providers" in provider_list
    assert "Mode Settings" in provider_list
    assert "Registered Types" in provider_list

    provider_select = screen.widgets["#remote-provider-select"]
    assert ("All providers", RemoteScreen.ALL_PROVIDERS_VALUE) in provider_select.options
    assert provider_select.value == RemoteScreen.ALL_PROVIDERS_VALUE
    assert screen.widgets["#remote-query-input"].focused is True

    remote_log = screen.widgets["#remote-log"]
    assert remote_log.lines == ["line1\nline2\n"]
    assert remote_log.ended is True


def test_focus_and_selection_sync() -> None:
    screen = _screen(context="query")
    screen._focus_default_input()
    assert screen.widgets["#remote-provider-select"].focused is False

    screen.app.current_context = "remote"
    screen.widgets["#remote-provider-select"].value = ""
    screen._focus_default_input()
    assert screen.widgets["#remote-provider-select"].focused is True

    screen.widgets["#remote-provider-select"].value = "wiki"
    screen._focus_default_input()
    assert screen.widgets["#remote-query-input"].focused is True

    screen.on_select_changed(SimpleNamespace(select=SimpleNamespace(id="remote-provider-select")))
    assert screen.widgets["#remote-weight-input"].value == "1.20"


def test_get_selected_provider_helpers_and_sync() -> None:
    screen = _screen()
    select = screen.widgets["#remote-provider-select"]
    select.value = RemoteScreen.ALL_PROVIDERS_VALUE
    assert screen._get_selected_provider_name() is None

    select.value = "wiki"
    assert screen._get_selected_provider_name() == "wiki"
    assert screen._get_provider_config("WIKI").name == "wiki"
    assert screen._get_provider_config("missing") is None

    screen._sync_provider_controls()
    assert screen.widgets["#remote-weight-input"].value == "1.20"


def test_set_provider_weight_paths() -> None:
    screen = _screen()
    select = screen.widgets["#remote-provider-select"]
    weight_input = screen.widgets["#remote-weight-input"]
    output = screen.widgets["#remote-results-output"]

    select.value = RemoteScreen.ALL_PROVIDERS_VALUE
    screen._set_provider_weight()
    assert "Select a specific provider" in output.last

    select.value = "wiki"
    weight_input.value = ""
    screen._set_provider_weight()
    assert "Enter a weight value" in output.last

    weight_input.value = "abc"
    screen._set_provider_weight()
    assert "Invalid weight value" in output.last

    weight_input.value = "-1"
    screen._set_provider_weight()
    assert "non-negative" in output.last

    weight_input.value = "2.5"
    screen._set_provider_weight()
    assert "weight set to 2.50" in output.last
    assert "2.50" in screen.widgets["#remote-provider-list"].last

    select.value = "missing"
    screen._set_provider_weight()
    assert "Provider not found" in output.last


def test_button_press_and_input_submit() -> None:
    screen = _screen()
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="remote-search-button")))
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="remote-refresh-button")))
    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="remote-weight-button")))
    assert len(screen.worker_calls) == 1

    screen.on_input_submitted(SimpleNamespace(input=SimpleNamespace(id="remote-query-input")))
    assert len(screen.worker_calls) == 2


def test_on_unmount_cancels_managed_lifecycle_entries() -> None:
    screen = _screen()
    cancelled: list[str] = []
    stopped: list[str] = []
    screen.app.cancel_managed_workers_for_owner = lambda **kwargs: cancelled.append(kwargs["reason"]) or {}  # type: ignore[attr-defined]
    screen.app.stop_managed_timers_for_owner = lambda **kwargs: stopped.append(kwargs["reason"]) or {}  # type: ignore[attr-defined]

    screen.on_unmount()

    assert cancelled == ["remote-unmount"]
    assert stopped == ["remote-unmount"]


def test_ensure_query_state(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    created = {}

    class _SessionState:
        def __init__(self, model, cost_tracker):
            created["model"] = model
            created["cost_tracker"] = cost_tracker

    monkeypatch.setattr("lsm.ui.tui.screens.remote.SessionState", _SessionState)
    monkeypatch.setattr("lsm.ui.tui.screens.remote.CostTracker", lambda: "tracker")

    screen._ensure_query_state()
    assert created["model"] == "gpt-test"
    assert created["cost_tracker"] == "tracker"


def test_run_search_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    select = screen.widgets["#remote-provider-select"]
    query_input = screen.widgets["#remote-query-input"]
    output = screen.widgets["#remote-results-output"]

    asyncio.run(screen._run_search())
    assert "Enter both a provider name and a query" in output.last

    async def _to_thread(fn, *args):
        return fn(*args)

    monkeypatch.setattr("asyncio.to_thread", _to_thread)

    select.value = RemoteScreen.ALL_PROVIDERS_VALUE
    screen.app.query_state = SimpleNamespace(model="gpt-test")
    query_input.value = "ethics"
    monkeypatch.setattr("lsm.ui.tui.screens.remote.run_remote_search_all", lambda q, cfg, state: f"all:{q}:{state.model}")
    asyncio.run(screen._run_search())
    assert output.last == "all:ethics:gpt-test"
    assert screen.is_loading is False

    select.value = "wiki"
    query_input.value = "logic"
    monkeypatch.setattr("lsm.ui.tui.screens.remote.run_remote_search", lambda p, q, cfg: f"{p}:{q}")
    asyncio.run(screen._run_search())
    assert output.last == "wiki:logic"

    def _boom(*_args):
        raise RuntimeError("fail")

    monkeypatch.setattr("lsm.ui.tui.screens.remote.run_remote_search", _boom)
    asyncio.run(screen._run_search())
    assert "Remote search failed: fail" in output.last


# -------------------------------------------------------------------------
# 5.7: Keyboard-first interaction parity
# -------------------------------------------------------------------------


def test_remote_bindings_include_search_and_refresh() -> None:
    """New keybindings for Ctrl+Enter and Ctrl+Shift+R exist on RemoteScreen."""
    binding_keys = {b.key for b in RemoteScreen.BINDINGS}
    assert "ctrl+enter" in binding_keys
    assert "ctrl+shift+r" in binding_keys


def test_remote_bindings_no_conflict_with_app() -> None:
    """Remote screen bindings must not conflict with app-level bindings."""
    from lsm.ui.tui.app import LSMApp

    remote_keys = {b.key for b in RemoteScreen.BINDINGS}
    app_keys = {b.key for b in LSMApp.BINDINGS}
    shared_ok = {"tab", "shift+tab"}
    conflicts = (remote_keys & app_keys) - shared_ok
    assert not conflicts, f"Key conflicts: {conflicts}"
