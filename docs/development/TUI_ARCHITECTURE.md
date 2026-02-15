# TUI Architecture Guide

This document defines the architectural conventions for the Local Second Mind TUI
(built on [Textual](https://textual.textualize.io/)). All TUI contributions must
follow these patterns.

## Screen Model

Tab content panels are **Widgets**, not Textual Screens. Only modals (Help,
UIErrorRecovery) use `ModalScreen`. The main `LSMApp` hosts a `TabbedContent`
whose panes contain the five tab widgets:

| Tab | Widget class | Module |
|-----|-------------|--------|
| Query | `QueryScreen(ManagedScreenMixin, Widget)` | `screens/query.py` |
| Ingest | `IngestScreen(Widget)` | `screens/ingest.py` |
| Remote | `RemoteScreen(Widget)` | `screens/remote.py` |
| Agents | `AgentsScreen(Widget)` | `screens/agents.py` |
| Settings | `SettingsScreen(Widget)` | `screens/settings.py` |

All tab widgets inherit from `ManagedScreenMixin` (`screens/base.py`) which
provides worker/timer lifecycle helpers. Screens that need custom shutdown
behavior override `_cancel_managed_workers` or `_stop_managed_timers`.

### Lifecycle

| Hook | Purpose |
|------|---------|
| `on_mount` | Initialize state, register workers/timers, focus default input |
| `on_unmount` | Cancel all owned workers and timers |
| `on_tabbed_content_tab_activated` | Lazy-load expensive state on first activation |

Every screen must clean up its workers and timers on unmount:

```python
def on_unmount(self) -> None:
    self._cancel_managed_workers(reason="screen-unmount")
    self._stop_managed_timers(reason="screen-unmount")
```

## State Management

### AppState

`lsm/ui/tui/state/app_state.py` provides a mutable typed container for
cross-screen state. Access it as `self.app.ui_state`.

- `set_active_context()` / `set_density_mode()` / `set_selected_agent_id()`
- `push_notification()` / `drain_notifications()`
- `snapshot()` returns a frozen `AppStateSnapshot` for safe read-only sharing

### SettingsViewModel

`lsm/ui/tui/state/settings_view_model.py` manages draft-vs-persisted config
state for the Settings screen. Actions (`update_field`, `add_item`,
`remove_item`, `rename_key`, `reset_tab`, `reset_all`, `save`) return typed
`SettingsActionResult` objects. Dirty tracking is per-field and per-tab.

### Rules

- **Do** mutate `AppState` via its setter methods.
- **Do** use `snapshot()` when passing state across threads.
- **Do not** store widget references in state objects.
- **Do not** mutate `AppState` from background threads. Post a Textual Message
  and handle it on the UI thread.

## Worker and Timer Lifecycle

All background workers and periodic timers are registered through the app-level
managed lifecycle API. Screens must never create raw threads or timers.

### Registration

```python
# Start a worker (cancel any existing with the same key)
self.app.start_managed_worker(
    owner="agents",
    key="running-refresh",
    worker=self.run_worker(coro, exclusive=True),
    timeout_s=30,
)

# Start a periodic timer
self.app.start_managed_timer(
    owner="agents",
    key="refresh-poll",
    interval=2.0,
    callback=self._on_refresh_tick,
)
```

### Cancellation

```python
# Cancel one worker
self.app.cancel_managed_worker(owner="agents", key="running-refresh")

# Cancel all workers/timers for a screen
self.app.cancel_managed_workers_for_owner("agents")
self.app.stop_managed_timers_for_owner("agents")

# App-level shutdown (called during quit)
self.app.shutdown_managed_workers(reason="app-quit")
self.app.shutdown_managed_timers(reason="app-quit")
```

### Data Structures

Workers and timers are keyed by `(owner_token, key_token)` and stored in
dictionaries protected by `threading.RLock`.

### Rules

- **Do** register every worker/timer through the app's managed API.
- **Do** cancel all owned workers and timers on screen unmount.
- **Do** specify `timeout_s` on workers for bounded join behavior.
- **Do not** use `threading.Thread` or `threading.Timer` directly.
- **Do not** call `time.sleep()` on the UI thread.

## Thread Safety

### UI Thread Access

Only the UI thread (the thread that created `LSMApp`) may mutate widgets. The
app caches its thread ID at init as `_ui_thread_id`.

### Posting from Background Threads

Use the Textual message queue to communicate from worker threads to the UI:

```python
# From any thread (safe)
self.app.post_ui_message(MyCustomMessage(data=result))

# Under the hood this uses call_from_thread()
```

### Agent Runtime Events

The agent runtime manager emits events from background threads. The app binds a
sink callback at mount and unbinds at quit:

```python
# Bind at mount
self._bind_agent_runtime_events()

# Sink callback (called from any thread)
def _on_agent_runtime_event_from_any_thread(self, event):
    self.post_ui_message(AgentRuntimeEvent(event))

# UI-thread handler
def on_agent_runtime_event(self, message):
    # Forward to agents screen
```

### Rules

- **Do** use `post_ui_message()` or `call_from_thread()` from background threads.
- **Do** use `Message` subclasses for all cross-thread communication.
- **Do not** call `query_one()`, `update()`, or any widget method from a
  background thread.
- **Do not** read reactive properties from background threads without a snapshot.

## Error Boundary

The app overrides `_handle_exception()` to intercept screen-level exceptions.
An exception is recoverable when its traceback includes a frame from
`lsm/ui/tui/screens/`.

Recovery flow:

1. Check `_is_recoverable_ui_exception()` against the traceback.
2. If recoverable, increment `_ui_error_count`.
3. Switch context to Query via `_set_safe_query_context()`.
4. Push `UIErrorRecoveryScreen` modal with error summary.
5. User can dismiss or press F12 to return to Query.

Non-screen exceptions (e.g., in `app.py` itself or in Textual internals) are
**not** recoverable and propagate normally.

### Rules

- **Do** let screen-level exceptions propagate naturally (the boundary catches
  them).
- **Do not** wrap entire screen methods in `try/except` just to suppress errors.
- **Do not** set `_recovering_ui_error = True` manually; the boundary manages it.

## Command Parsing

Slash commands are parsed through shared helpers in
`lsm/ui/helpers/commands/common.py`:

- `parse_slash_command(text)` returns a frozen `ParsedCommand` with `.cmd`,
  `.parts`, `.text`.
- `tokenize_command(text, use_shlex=False)` splits arguments.
- `normalize_argument()` / `normalize_arguments()` for input cleanup.
- `parse_on_off_value()` for boolean toggle parsing.
- `format_command_error()` for consistent error formatting.

Commands are case-insensitive. The `.cmd` field is always lowercased.

### Autocomplete

`lsm/ui/tui/completions.py` maintains `QUERY_COMMANDS`, `INGEST_COMMANDS`, and
per-command value sets. These must stay aligned with the implemented command
handlers.

## Keybinding Conventions

| Scope | Keys | Notes |
|-------|------|-------|
| App-level tab switching | Ctrl+Q/N/R/G/S | Do not reuse in screens |
| Help | F1 | Global |
| Safe return | F12 | Global |
| Screen function keys | F2-F11 | Per-screen (Settings tabs, Agents actions) |
| Screen Ctrl combos | Ctrl+B/T/E/O/L/I | Per-screen actions |
| Shared navigation | Tab, Shift+Tab | Field navigation |
| Refresh | Ctrl+Shift+R | Convention across multiple screens |

### Rules

- **Do** check for conflicts with app-level bindings before adding screen
  bindings.
- **Do** document new keybindings in `lsm/ui/tui/screens/help.py` context
  shortcuts.
- **Do not** reuse app-level tab-switch keys in screen bindings.

## CSS Organization

Styles are split under `lsm/ui/tui/styles/`:

| File | Scope |
|------|-------|
| `base.tcss` | Layout, shared classes |
| `widgets.tcss` | Reusable widget styles |
| `query.tcss` | Query screen |
| `ingest.tcss` | Ingest screen |
| `settings.tcss` | Settings screen |
| `remote.tcss` | Remote screen |
| `agents.tcss` | Agents screen |

`LSMApp.CSS_PATH` loads all files in deterministic order. When adding a new
screen, add a corresponding CSS file and register it in the load list.

## Testing

### Test Doubles

TUI tests use lightweight fake widget doubles instead of mounting real Textual
apps:

```python
class _TestableScreen(TargetScreen):
    def __init__(self, app):
        super().__init__()
        self._test_app = app
        self.widgets = {}

    @property
    def app(self):
        return self._test_app

    def query_one(self, selector, _cls=None):
        return self.widgets[selector]
```

### Test Categories

- **Contract tests** verify parsing grammar and command validation without
  runtime.
- **Structure tests** verify widget composition and CSS class presence.
- **Behavior tests** verify interaction flows through fake doubles.
- **Binding tests** verify keybinding sets and conflict-freedom.

### Rules

- **Do** use fake widget doubles for unit tests.
- **Do** test keybinding sets for conflicts with app-level bindings.
- **Do** test command handlers with mocked runtime dependencies.
- **Do not** mount a real Textual app in unit tests (use integration tests for
  that).
- **Do not** write stub tests that assert `True` without exercising real logic.
