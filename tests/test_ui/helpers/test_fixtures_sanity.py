"""Sanity tests for shared TUI test fixtures.

Verifies the reusable test doubles work correctly so downstream tests
can depend on them confidently.
"""

from __future__ import annotations

from tests.test_ui.tui.fixtures import (
    FakeStatic,
    FakeInput,
    FakeSelect,
    FakeRichLog,
    FakeButton,
    create_fake_app,
)


class TestFakeWidgets:
    def test_static_update(self) -> None:
        w = FakeStatic("initial")
        assert w.last == "initial"
        w.update("new")
        assert w.last == "new"

    def test_input_focus(self) -> None:
        w = FakeInput("val", "my-input")
        assert w.value == "val"
        assert w.id == "my-input"
        assert not w.focused
        w.focus()
        assert w.focused

    def test_select_options(self) -> None:
        w = FakeSelect("v", "my-select")
        w.set_options([("a", "1"), ("b", "2")])
        assert len(w.options) == 2
        w.focus()
        assert w.focused

    def test_rich_log_write(self) -> None:
        w = FakeRichLog()
        w.write("line1")
        w.write("line2")
        assert w.lines == ["line1", "line2"]
        w.scroll_end()
        assert w.ended

    def test_button(self) -> None:
        w = FakeButton("Click", "btn-1")
        assert w.label == "Click"
        assert w.id == "btn-1"


class TestFakeApp:
    def test_default(self) -> None:
        app = create_fake_app()
        assert app.current_context == "query"
        assert app.config.query.mode == "grounded"
        assert app.config.llm.get_query_config().model == "gpt-test"

    def test_custom_context(self) -> None:
        app = create_fake_app(current_context="ingest")
        assert app.current_context == "ingest"

    def test_custom_config(self) -> None:
        from types import SimpleNamespace
        cfg = SimpleNamespace(query=SimpleNamespace(mode="insight"))
        app = create_fake_app(config=cfg)
        assert app.config.query.mode == "insight"
