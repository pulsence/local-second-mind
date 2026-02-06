from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

import lsm.ui.tui.screens.ingest as ingest_screen_mod
from lsm.ui.tui.commands.ingest import CommandResult


class _Static:
    def __init__(self) -> None:
        self.value = ""
        self.styles = SimpleNamespace(display="block")

    def update(self, text: str) -> None:
        self.value = text


class _ProgressBar:
    def __init__(self) -> None:
        self.total = None
        self.progress = None

    def update(self, total, progress) -> None:
        self.total = total
        self.progress = progress


class _TreeRoot:
    def __init__(self) -> None:
        self.label = ""
        self.children = []
        self.expanded = False

    def set_label(self, label: str) -> None:
        self.label = label

    def remove_children(self) -> None:
        self.children = []

    def add(self, label: str):
        node = SimpleNamespace(label=label, data=None, add=self.add)
        self.children.append(node)
        return node

    def expand(self) -> None:
        self.expanded = True


class _Tree:
    def __init__(self) -> None:
        self.styles = SimpleNamespace(display="none")
        self.root = _TreeRoot()
        self.focused = False
        self.id = "ingest-explore-tree"

    def focus(self) -> None:
        self.focused = True


class _Input:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _TestableIngestScreen(ingest_screen_mod.IngestScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self.widgets = {}

    @property
    def app(self):  # type: ignore[override]
        return self._test_app

    def query_one(self, selector, _cls=None):  # type: ignore[override]
        if selector in self.widgets:
            return self.widgets[selector]
        raise KeyError(selector)


def _screen():
    app = SimpleNamespace(
        ingest_provider=SimpleNamespace(count=lambda: 3),
        config=SimpleNamespace(collection="kb", vectordb=SimpleNamespace(provider="chromadb")),
        call_from_thread=lambda fn: fn(),
        update_chunk_count=lambda n: None,
        _async_init_ingest_context=lambda: None,
        exit=lambda: None,
    )
    screen = _TestableIngestScreen(app)
    screen.widgets["#ingest-output"] = _Static()
    screen.widgets["#ingest-explore-tree"] = _Tree()
    screen.widgets["#ingest-command-status"] = _Static()
    screen.widgets["#command-progress"] = _ProgressBar()
    screen.widgets["#ingest-selection-stats"] = _Static()
    screen.widgets["#stats-content"] = _Static()
    screen.widgets["#ingest-command-input"] = _Input()
    return screen


def test_output_mode_and_progress_updates() -> None:
    screen = _screen()
    screen._set_output_mode("tree")
    assert screen.widgets["#ingest-output"].styles.display == "none"
    assert screen.widgets["#ingest-explore-tree"].styles.display == "block"

    screen._update_output_text("hello")
    assert screen.widgets["#ingest-output"].value == "hello"
    screen._set_command_status("running")
    assert screen.widgets["#ingest-command-status"].value == "running"

    screen._update_command_progress(None, None)
    assert screen.widgets["#command-progress"].total == 1
    screen._update_command_progress(3, 10)
    assert screen.widgets["#command-progress"].progress == 3


def test_execute_ingest_command_action_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()
    cfg = screen.app.config
    provider = screen.app.ingest_provider

    monkeypatch.setattr(
        ingest_screen_mod,
        "handle_ingest_command",
        lambda *args, **kwargs: CommandResult(output="ok", action="build_confirm", action_data={"config": cfg}),
    )
    monkeypatch.setattr(ingest_screen_mod, "run_build", lambda *args, **kwargs: "built")
    result, prompt, _ = screen._execute_ingest_command("x", deque(["yes"]), None, None)
    assert "built" in result.output
    assert prompt is None

    result2, prompt2, _ = screen._execute_ingest_command("x", deque([]), None, None)
    assert prompt2 is not None

    monkeypatch.setattr(
        ingest_screen_mod,
        "handle_ingest_command",
        lambda *args, **kwargs: CommandResult(output="ok", action="wipe_confirm", action_data={"config": cfg}),
    )
    monkeypatch.setattr(ingest_screen_mod, "run_wipe", lambda *args, **kwargs: "wiped")
    result3, _, _ = screen._execute_ingest_command("x", deque(["DELETE", "yes"]), None, None)
    assert "wiped" in result3.output

    monkeypatch.setattr(
        ingest_screen_mod,
        "handle_ingest_command",
        lambda *args, **kwargs: CommandResult(output="ok", action="tag_confirm", action_data={"config": cfg, "provider": provider}),
    )
    monkeypatch.setattr(ingest_screen_mod, "run_tag", lambda *args, **kwargs: "tagged")
    result4, _, _ = screen._execute_ingest_command("x", deque(["yes"]), None, None)
    assert "tagged" in result4.output


def test_execute_ingest_command_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_screen_mod, "handle_ingest_command", _boom)
    result, prompt, _ = screen._execute_ingest_command("x", deque(), None, None)
    assert result.handled is False
    assert "Error: boom" in result.output
    assert prompt is None


def test_run_ingest_command_and_refresh_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _screen()

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(
        screen,
        "_execute_ingest_command",
        lambda *args, **kwargs: (CommandResult(output="done", should_exit=False), None, []),
    )
    out, prompt = asyncio.run(screen._run_ingest_command("/stats", []))
    assert out == "done"
    assert prompt is None

    asyncio.run(screen._refresh_stats())
    assert "Collection: kb" in screen.widgets["#stats-content"].value


def test_tree_selection_and_actions() -> None:
    screen = _screen()
    tree = screen.widgets["#ingest-explore-tree"]
    root = {"file_count": 1, "chunk_count": 2, "children": {}, "files": {"a.md": 2}}
    screen._render_explore_tree(root, "docs")
    assert tree.root.expanded is True
    assert tree.focused is True

    event = SimpleNamespace(node=SimpleNamespace(tree=SimpleNamespace(id="ingest-explore-tree"), data={"type": "file", "path": "/a.md", "chunk_count": 2}))
    screen.on_tree_node_selected(event)
    assert "/a.md" in screen.widgets["#ingest-selection-stats"].value

    event2 = SimpleNamespace(node=SimpleNamespace(tree=SimpleNamespace(id="ingest-explore-tree"), data={"type": "dir", "path": "/", "file_count": 1, "chunk_count": 2}))
    screen.on_tree_node_highlighted(event2)
    assert "1 files" in screen.widgets["#ingest-selection-stats"].value

    called = {}

    def _run_worker(coro, exclusive=True):
        called["ran"] = True
        coro.close()

    screen.run_worker = _run_worker  # type: ignore[method-assign]
    screen.action_run_build()
    assert called["ran"] is True
    screen.action_clear_input()
    assert screen.widgets["#ingest-command-input"].cleared is True
