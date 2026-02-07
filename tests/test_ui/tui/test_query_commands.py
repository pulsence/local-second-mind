from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from lsm.query.session import Candidate
from lsm.ui.tui.screens.query import QueryScreen


class _Tracker:
    def __init__(self) -> None:
        self.budget_limit = None
        self.exported_path = None

    def export_csv(self, path: Path) -> None:
        self.exported_path = path


class _ProviderCfg:
    def __init__(self, provider_name):
        self.provider_name = provider_name


class _LLM:
    def __init__(self) -> None:
        self._feature_cfg = SimpleNamespace(provider="openai", model="gpt-test")
        self.last_set = None

    def get_feature_provider_map(self):
        return {"query": "openai", "tagging": "openai", "ranking": "openai"}

    def get_query_config(self):
        return self._feature_cfg

    def get_tagging_config(self):
        return SimpleNamespace(provider="openai", model="gpt-tag")

    def get_ranking_config(self):
        return SimpleNamespace(provider="openai", model="gpt-rank")

    def get_provider_names(self):
        return ["openai"]

    def get_provider_by_name(self, _name):
        return _ProviderCfg(_name)

    def resolve_any_for_provider(self, _name):
        return self._feature_cfg

    def set_feature_selection(self, feature: str, provider: str, model: str):
        self.last_set = (feature, provider, model)


class _FakeApp:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            llm=_LLM(),
            vectordb=SimpleNamespace(provider="chromadb", collection="kb"),
            query=SimpleNamespace(mode="grounded", path_contains=None, ext_allow=None, ext_deny=None),
            notes=SimpleNamespace(enabled=True),
            modes={
                "grounded": SimpleNamespace(
                    synthesis_style="grounded",
                    source_policy=SimpleNamespace(
                        local=SimpleNamespace(k=12),
                        remote=SimpleNamespace(enabled=False, max_results=5, rank_strategy="weighted", remote_providers=None),
                        model_knowledge=SimpleNamespace(enabled=False),
                    ),
                )
            },
            get_mode_config=lambda mode=None: SimpleNamespace(
                synthesis_style="grounded",
                source_policy=SimpleNamespace(
                    local=SimpleNamespace(k=12),
                    remote=SimpleNamespace(enabled=False, max_results=5, rank_strategy="weighted", remote_providers=None),
                    model_knowledge=SimpleNamespace(enabled=False),
                ),
            ),
        )
        self.query_state = SimpleNamespace(
            format_debug=lambda: "debug",
            format_costs=lambda: "costs",
            cost_tracker=_Tracker(),
            last_label_to_candidate={},
            last_question=None,
            last_answer="",
            last_local_sources_for_notes=[],
            last_remote_sources=[],
            pinned_chunks=[],
            path_contains=None,
            ext_allow=None,
            ext_deny=None,
            model="gpt-test",
        )
        self.query_provider = object()
        self.query_embedder = object()
        self.current_mode = "grounded"
        self.notified = []
        self.exited = False
        self._tui_log_buffer = ["a", "b"]

    def notify(self, msg: str, severity: str = "info"):
        self.notified.append((msg, severity))

    def exit(self):
        self.exited = True

    def call_from_thread(self, fn):
        fn()

    def query_one(self, selector, _cls=None):
        if selector == "#main-status-bar":
            return SimpleNamespace(provider_status="")
        raise KeyError(selector)


class _TestableQueryScreen(QueryScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self._widgets = {}
        self.messages = []

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        if selector in self._widgets:
            return self._widgets[selector]
        return self.app.query_one(selector, _cls)

    def set_widget(self, selector: str, widget):
        self._widgets[selector] = widget

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()

    def _show_message(self, message: str, preserve_candidates: bool = True) -> None:
        self.messages.append((message, preserve_candidates))


def _screen() -> _TestableQueryScreen:
    return _TestableQueryScreen(_FakeApp())


def test_format_helpers_and_models(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    assert "query: openai/gpt-test" in screen._format_model_selection()

    monkeypatch.setattr("lsm.ui.tui.screens.query.create_provider", lambda cfg: SimpleNamespace(list_models=lambda: ["m2", "m1"]))
    out = screen._format_models("/models")
    assert "openai:" in out
    assert "m1" in out and "m2" in out

    out2 = screen._format_models("/models missing")
    assert "Provider not found in config" in out2


def test_execute_command_basic_and_costs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    screen = _screen()

    assert screen._execute_query_command("hello").handled is False
    assert screen._execute_query_command("/exit").should_exit is True
    assert "QUERY COMMANDS" in screen._execute_query_command("/help").output
    assert screen._execute_query_command("/debug").output == "debug"
    assert screen._execute_query_command("/costs").output == "costs"

    export_path = tmp_path / "costs.csv"
    out = screen._execute_query_command(f"/costs export {export_path}").output
    assert "Cost data exported to" in out

    def _boom(_p):
        raise RuntimeError("x")

    screen.app.query_state.cost_tracker.export_csv = _boom
    assert "Failed to export costs" in screen._execute_query_command("/costs export bad.csv").output


def test_execute_budget_and_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    assert "No budget set" in screen._execute_query_command("/budget").output
    assert "Usage" in screen._execute_query_command("/budget set").output
    assert "Invalid budget amount" in screen._execute_query_command("/budget set nope").output
    assert "Budget limit set to: $2.5000" in screen._execute_query_command("/budget set 2.5").output
    assert "Budget limit: $2.5000" in screen._execute_query_command("/budget").output

    monkeypatch.setattr("lsm.ui.tui.screens.query.estimate_query_cost", lambda *args, **kwargs: 1.2345)
    assert "Estimated cost: $1.2345" in screen._execute_query_command("/cost-estimate test query").output


def test_execute_model_and_mode_commands() -> None:
    screen = _screen()
    assert "Unknown task" in screen._execute_query_command("/model invalid p m").output
    ok = screen._execute_query_command("/model query openai gpt-x").output
    assert "Model set: query = openai/gpt-x" in ok
    assert screen.app.query_state.model == "gpt-x"

    show_mode = screen._execute_query_command("/mode").output
    assert "Current mode: grounded" in show_mode
    assert "Available modes" in show_mode

    bad_mode = screen._execute_query_command("/mode missing").output
    assert "Mode not found" in bad_mode

    mode_set = screen._execute_query_command("/mode set notes off").output
    assert "set to: off" in mode_set
    assert screen.app.config.notes.enabled is False


def test_execute_export_and_load(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    assert "Format must be 'bibtex' or 'zotero'" in screen._execute_query_command("/export-citations bad").output
    assert "No last query sources available" in screen._execute_query_command("/export-citations bibtex").output

    cand = Candidate(cid="1", text="t", meta={"source_path": "/a.md", "source_name": "a.md", "chunk_index": 0, "ext": ".md"}, distance=0.1)
    screen.app.query_state.last_label_to_candidate = {"S1": cand}
    monkeypatch.setattr("lsm.ui.tui.screens.query.export_citations_from_sources", lambda sources, fmt: Path("out.bib"))
    assert "Citations exported to: out.bib" in screen._execute_query_command("/export-citations bibtex").output

    assert "Usage: /load <file_path>" in screen._execute_query_command("/load").output
    assert "Cleared all pinned chunks" in screen._execute_query_command("/load clear").output

    fake_chroma_empty = SimpleNamespace(get=lambda **kwargs: {"ids": []})
    monkeypatch.setattr("lsm.ui.tui.screens.query.require_chroma_collection", lambda provider, why: fake_chroma_empty)
    assert "No chunks found for path" in screen._execute_query_command("/load /missing.md").output

    fake_chroma = SimpleNamespace(get=lambda **kwargs: {"ids": ["c1", "c2"]})
    monkeypatch.setattr("lsm.ui.tui.screens.query.require_chroma_collection", lambda provider, why: fake_chroma)
    out = screen._execute_query_command("/load /ok.md").output
    assert "Pinned 2 chunks" in out
    assert len(screen.app.query_state.pinned_chunks) == 2


def test_execute_show_open_set_clear_and_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    assert "Usage: /show" in screen._execute_query_command("/show").output
    assert "Usage: /open" in screen._execute_query_command("/open").output
    assert "Usage:" in screen._execute_query_command("/set").output
    assert "Usage: /clear" in screen._execute_query_command("/clear").output

    cand = SimpleNamespace(
        meta={"source_path": "/doc.md"},
        format=lambda label, expanded=False: f"{label}:{expanded}",
    )
    screen.app.query_state.last_label_to_candidate = {"S1": cand}
    assert "S1:False" in screen._execute_query_command("/show s1").output
    assert "S1:True" in screen._execute_query_command("/expand s1").output

    monkeypatch.setattr("lsm.ui.tui.screens.query.open_file", lambda path: True)
    assert "Opened: /doc.md" in screen._execute_query_command("/open S1").output
    monkeypatch.setattr("lsm.ui.tui.screens.query.open_file", lambda path: False)
    assert "Failed to open file" in screen._execute_query_command("/open S1").output

    assert "path_contains set to: one" in screen._execute_query_command("/set path_contains one").output
    assert "ext_allow set to" in screen._execute_query_command("/set ext_allow .md .txt").output
    assert "Unknown filter key" in screen._execute_query_command("/set bad x").output
    assert "path_contains cleared" in screen._execute_query_command("/clear path_contains").output

    async def _fake_to_thread(fn, arg):
        return fn(arg)

    monkeypatch.setattr("asyncio.to_thread", _fake_to_thread)
    out = asyncio.run(screen._run_query_command("/quit"))
    assert out == ""
    assert screen.app.exited is True


def test_show_citation_expand_and_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    panel = SimpleNamespace(
        get_candidate=lambda idx: Candidate(
            cid="1",
            text="x" * 600,
            meta={"source_path": "/doc.md", "chunk_index": 2},
            distance=0.1234,
        )
        if idx == 1
        else None,
        expand_citation=lambda idx: setattr(panel, "expanded", idx),
    )
    cmd_input = SimpleNamespace(value="hi", clear=lambda: setattr(cmd_input, "value", ""))
    log = SimpleNamespace(clear=lambda: setattr(log, "cleared", True), write=lambda m: None, scroll_end=lambda: setattr(log, "ended", True))
    screen.set_widget("#query-results-panel", panel)
    screen.set_widget("#query-command-input", cmd_input)
    screen.set_widget("#query-log", log)

    screen._show_citation("S1")
    assert any("Citation S1" in m[0] for m in screen.messages)
    screen._show_citation("bad")
    assert any("Invalid citation format" in m[0] for m in screen.messages)

    screen._expand_citation("S1")
    assert getattr(panel, "expanded", None) == 1

    screen.selected_citation = None
    screen.action_expand_citation()
    assert screen.app.notified[-1][0] == "No citation selected"

    screen.action_submit_query()
    assert cmd_input.value == ""
    screen.action_clear_input()
    assert cmd_input.value == ""
    screen.action_refresh_logs()
    assert getattr(log, "cleared", False) is True
