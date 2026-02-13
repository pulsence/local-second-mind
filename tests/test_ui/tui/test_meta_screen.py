from __future__ import annotations

from types import SimpleNamespace

from lsm.ui.tui.screens.agents import AgentsScreen


class _Input:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id

    def focus(self) -> None:
        return None


class _Select:
    def __init__(self, value: str = "", widget_id: str = "") -> None:
        self.value = value
        self.id = widget_id
        self.options = []

    def set_options(self, options) -> None:
        self.options = options


class _DataTable:
    def __init__(self, widget_id: str = "") -> None:
        self.id = widget_id
        self.columns = []
        self.rows = []
        self.cursor_row = 0
        self.cursor_type = None
        self.zebra_stripes = False

    @property
    def column_count(self) -> int:
        return len(self.columns)

    def add_columns(self, *columns) -> None:
        self.columns.extend(columns)

    def clear(self, *, columns: bool = False) -> None:
        self.rows = []
        if columns:
            self.columns = []

    def add_row(self, *row_values) -> None:
        self.rows.append(tuple(row_values))

    def move_cursor(self, *, row: int, column: int = 0) -> None:
        _ = column
        self.cursor_row = row

    @property
    def cursor_coordinate(self):
        return SimpleNamespace(row=self.cursor_row, column=0)


class _Static:
    def __init__(self) -> None:
        self.last = ""

    def update(self, message: str) -> None:
        self.last = message


class _RichLog:
    def __init__(self) -> None:
        self.lines = []

    def write(self, message: str) -> None:
        self.lines.append(message)


class _Manager:
    def __init__(self) -> None:
        self.snapshot = {
            "available": True,
            "status": "completed",
            "thread_alive": False,
            "goal": "Compose weekly brief",
            "execution_order": ["research_1", "writing_1"],
            "tasks": [
                {
                    "id": "research_1",
                    "agent_name": "research",
                    "status": "completed",
                    "depends_on": [],
                },
                {
                    "id": "writing_1",
                    "agent_name": "writing",
                    "status": "completed",
                    "depends_on": ["research_1"],
                },
            ],
            "task_runs": [
                {
                    "task_id": "research_1",
                    "agent_name": "research",
                    "status": "completed",
                    "sub_agent_dir": "/tmp/meta/sub_agents/research_001",
                    "artifacts": [],
                    "error": None,
                },
                {
                    "task_id": "writing_1",
                    "agent_name": "writing",
                    "status": "completed",
                    "sub_agent_dir": "/tmp/meta/sub_agents/writing_001",
                    "artifacts": ["/tmp/meta/final_result.md"],
                    "error": None,
                },
            ],
            "artifacts": [
                "/tmp/meta/final_result.md",
                "/tmp/meta/meta_log.md",
            ],
            "final_result_path": "/tmp/meta/final_result.md",
            "meta_log_path": "/tmp/meta/meta_log.md",
        }

    def get_meta_snapshot(self):
        return dict(self.snapshot)

    def meta_log(self):
        return "# Meta Log\n\nTask trace...\n"

    def list_schedules(self, app):
        _ = app
        return []

    def get_memory_candidates(self, app, status="pending", limit=200):
        _ = app, status, limit
        return []


class _TestableAgentsScreen(AgentsScreen):
    def __init__(self, app) -> None:
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

    def call_after_refresh(self, fn):  # type: ignore[override]
        fn()


def _screen():
    app = SimpleNamespace(current_context="agents", config=SimpleNamespace())
    screen = _TestableAgentsScreen(app)
    screen.widgets["#agents-select"] = _Select("", "agents-select")
    screen.widgets["#agents-topic-input"] = _Input("", "agents-topic-input")
    screen.widgets["#agents-status-output"] = _Static()
    screen.widgets["#agents-log"] = _RichLog()

    screen.widgets["#agents-meta-task-table"] = _DataTable("agents-meta-task-table")
    screen.widgets["#agents-meta-runs-table"] = _DataTable("agents-meta-runs-table")
    screen.widgets["#agents-meta-output"] = _Static()
    screen.widgets["#agents-meta-artifacts-output"] = _Static()

    screen.widgets["#agents-schedule-table"] = _DataTable("agents-schedule-table")
    screen.widgets["#agents-schedule-agent-input"] = _Input("", "agents-schedule-agent-input")
    screen.widgets["#agents-schedule-interval-input"] = _Input("", "agents-schedule-interval-input")
    screen.widgets["#agents-schedule-params-input"] = _Input("", "agents-schedule-params-input")
    screen.widgets["#agents-schedule-concurrency-select"] = _Select("skip", "agents-schedule-concurrency-select")
    screen.widgets["#agents-schedule-confirmation-select"] = _Select("auto", "agents-schedule-confirmation-select")
    screen.widgets["#agents-schedule-output"] = _Static()

    screen.widgets["#agents-memory-select"] = _Select("", "agents-memory-select")
    screen.widgets["#agents-memory-ttl-input"] = _Input("", "agents-memory-ttl-input")
    screen.widgets["#agents-memory-output"] = _Static()
    return screen


def test_meta_panel_renders_task_graph_runs_and_artifacts(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["meta", "research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    task_table = screen.widgets["#agents-meta-task-table"]
    runs_table = screen.widgets["#agents-meta-runs-table"]
    assert len(task_table.rows) == 2
    assert len(runs_table.rows) == 2
    assert "tasks=2 completed=2 failed=0" in screen.widgets["#agents-meta-output"].last
    assert "final_result: /tmp/meta/final_result.md" in screen.widgets[
        "#agents-meta-artifacts-output"
    ].last


def test_meta_panel_handles_no_active_meta(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    manager.snapshot["available"] = False
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["meta", "research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )

    screen.on_mount()
    assert "No active meta-agent." in screen.widgets["#agents-meta-output"].last


def test_meta_log_button_writes_to_log(monkeypatch) -> None:
    screen = _screen()
    manager = _Manager()
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.AgentRegistry",
        lambda: SimpleNamespace(list_agents=lambda: ["meta", "research"]),
    )
    monkeypatch.setattr(
        "lsm.ui.tui.screens.agents.get_agent_runtime_manager",
        lambda: manager,
    )
    screen.on_mount()

    screen.on_button_pressed(SimpleNamespace(button=SimpleNamespace(id="agents-meta-log-button")))
    assert any("Meta Log" in line for line in screen.widgets["#agents-log"].lines)
