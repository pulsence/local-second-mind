# Changelog

All notable changes to Local Second Mind are documented here.

## 0.7.1 (unreleased)

### Added

- `PhaseResult` dataclass in `lsm.agents.phase` for phase execution results (no token/cost data).
- `AgentHarness.run_bounded()` method with multi-context support via `context_label`, `continue_context`, and tool-only mode via `direct_tool_calls`.
- `BaseAgent._run_phase()` method for bounded phase execution with context management.
- `BaseAgent._reset_harness()` helper to reset harness between run() calls.
- BaseAgent workspace accessor methods: `_workspace_root()`, `_artifacts_dir()`, `_logs_dir()`, `_memory_dir()`, `_artifact_filename()`.
- `acknowledged_timeout_seconds` config option in `agents.interaction` for two-phase interaction timeout behavior (default `0` = infinite wait once acknowledged).
- Two-phase timeout in `InteractionChannel`: unacknowledged requests time out after `timeout_seconds`; acknowledged requests wait indefinitely (or for `acknowledged_timeout_seconds` if configured).
- `acknowledge_interaction(agent_id, request_id)` method in `AgentRuntimeManager` to forward acknowledgment signals to interaction channels.
- Automatic acknowledgment from TUI when pending interactions are rendered in the Agents screen.
- Automatic acknowledgment from shell `/agent interact` command before displaying interaction prompts.

### Changed

- `InteractionChannel.post_request()` now uses a polling loop with two-phase timeout logic instead of a single `event.wait()` call.
- `_acknowledged_interaction_ids` tracking in TUI Agents screen to prevent duplicate acknowledgment signals.
- Replaced `query_embeddings` tool with `query_knowledge_base` tool that uses the full query pipeline (embedding search + reranking + LLM synthesis).
- `AgentHarness.run_bounded()` now logs LLM responses and tool executions to `state.log_entries` for consistency with the full `run()` method.
- `GeneralAgent`, `LibrarianAgent`, and `ManuscriptEditorAgent` migrated from direct `AgentHarness` instantiation to `self._run_phase()`. File output paths updated to use `self._artifacts_dir()` workspace accessor.
- `ResearchAgent` migrated to `_run_phase()`: per-subtopic `context_label` usage, `query_knowledge_base` integration in the RESEARCH phase, improved subtopic and suggestion logging (names appear in logs, not just counts), and artifact output via `_artifacts_dir()`.
- `SynthesisAgent` migrated to `_run_phase()`: three-phase workflow (PLAN → EVIDENCE → SYNTHESIZE) via `_run_phase()`, file output via `_artifacts_dir()`.
- `CuratorAgent` migrated to `_run_phase()`: scope selection and recommendation generation use `_run_phase()` instead of direct `provider.synthesize()` calls; `CURATOR_SYSTEM_PROMPT` constant added; manual token tracking removed.

### Removed

- `QueryEmbeddingsTool` from `lsm.agents.tools` - replaced by `QueryKnowledgeBaseTool`.

## 0.7.0 - 2026-02-25

### Added

- Tiered LLM configuration via `llms.tiers` for agent/tool model selection.
- OpenRouter LLM provider with model fallback routing, prompt caching markers, and usage tracking.
- BaseOAIProvider shared OAI-PMH parsing utilities with consolidated client/parser logic.
- MCP host support with `global.mcp_servers` for registering external tool servers.
- RSS/Atom remote provider with feed caching, seen-item tracking, and normalized RemoteResult output.
- Remote provider sub-packages with structured output validation for all remote sources.
- Scholarly discovery chain (OpenAlex → Crossref → Unpaywall → CORE) with preconfigured `remote.chains` enablement and full-text downloads.
- New academic providers: Unpaywall, PubMed, SSRN, PhilArchive, Project MUSE.
- New cultural heritage providers: Archive.org, DPLA, Library of Congress, Smithsonian, Met Museum, Rijksmuseum, IIIF, Wikidata, Perseus CTS.
- New news providers: NYTimes, The Guardian, GDELT, NewsAPI.
- OAuth2 infrastructure for user-authorized providers with encrypted token storage and refresh support.
- Email providers for Gmail, Microsoft Graph Mail, and IMAP/SMTP.
- Calendar providers for Google Calendar, Microsoft Graph Calendar, and CalDAV.
- Communication assistants for email, calendar, and news summaries with approval-gated actions.
- Parallel task graph planning: `parallel_group` nodes in `TaskGraph` with dependency gate enforcement, deterministic ordering via topological sort, and graph serialization round-trips.
- Parallel execution engine in `MetaAgent`: `ThreadPoolExecutor` with `max_concurrent` resource limits, sandbox monotonicity (child sandboxes are strict subsets of parent), and deterministic artifact merge sorted by `(agent_name, task_order)`.
- General meta-agent (`meta`) with system prompt, keyword-based sub-agent selection, and `final_result.md` artifact emission via LLM synthesis with structured fallback.
- Assistant meta-agent (`assistant_meta`) with validation passes over sub-agent outputs, findings detection (errors, TODOs, permission denials), and `assistant_meta_summary.json`/`.md` action recommendations.
- Agent/tool tier declarations with tier-based model selection in agent harness runs.
- Benchmark harness for agent/tool regression tracking with baseline file-operation tasks in `tests/benchmarks/`.
- File graphing system with unified graph schema, deterministic IDs, content-hash caching, and line-hash metadata.
- Code/text/PDF/HTML graphers with section-aware spans, fixtures, and coverage.
- Graph-aware read and metadata tooling with section selection and optional graph output.
- Graph-aware file tooling: `find_file`, `find_section`, and line-hash `edit_file` for structural navigation and edits.
- Structured `read_file` output with section selection, outlines, and optional line hashes.
- Benchmark comparisons for advanced tooling with recorded Phase 4 results in `tests/benchmarks/results/phase_4.md`.
- Docker runner improvements with read/write mounts, path translation, and scrubbed environment forwarding.
- WSL2 runner support for executing tools on Windows hosts with path translation and availability checks.
- New `bash` and `powershell` execution tools with sandbox-enforced allow/deny lists and command chaining guards.
- Core agent catalog entries (`general`, `librarian`, `assistant`, `manuscript_editor`) with registry metadata exposure and UI listing updates.
- General agent tool loop with custom system prompt, summary artifact output, and guardrail coverage tests.
- Librarian agent workflow for embeddings-driven idea graphs with memory proposal artifacts.
- Assistant agent summaries aggregating run activity with memory candidate proposals.
- Manuscript editor section-level revisions with revision logs and finalized manuscript artifacts.
- Standard per-agent workspace layout (`logs/`, `artifacts/`, `memory/`) with file tools defaulting to the agent workspace root.
- Native tool-calling support with provider function-calling APIs (OpenAI, Anthropic, Gemini) and prompt-schema fallback for providers without tool APIs.
- `agents.interaction.auto_continue` to auto-respond to `ask_user` prompts with a continuation message.

### Changed

- Restructured built-in agents into academic/productivity/meta subpackages with registry theme/category metadata and grouped agent lists in shell/TUI.
- `source_map` now returns structural outlines derived from file graphs instead of snippet lists.

## 0.6.0 - 2026-02-19

### Added

- TUI density mode support with `auto`, `compact`, and `comfortable` behavior:
  - persisted `global.tui_density_mode` config support in loader/serializer and `GlobalConfig`
  - `/ui density` query command for runtime status and mode switching
  - auto-density terminal sizing with resize re-evaluation and debounce/hysteresis behavior
  - narrow-terminal responsive layout fallbacks for query, remote, agents, and ingest panes
- Dedicated agents screen styling for layout consistency and small-terminal usability:
  - explicit panel/layout rules for `#agents-layout`, `#agents-top`, `#agents-left`, and `#agents-log-panel`
  - dedicated section, button-group, and DataTable styling for status/meta/schedule/memory panels
- New TUI style regression coverage for density behavior and agents layout:
  - `tests/test_ui/tui/test_compact_layout.py`
  - `tests/test_ui/tui/test_agents_layout_css.py`
- Settings state/view-model foundation and global UI state:
  - new `lsm/ui/tui/state/settings_view_model.py` with draft vs persisted config buckets, typed actions (`update_field`, `add_item`, `remove_item`, `rename_key`, `reset_tab`, `reset_all`, `save`), dirty tracking, and validation error state
  - new `lsm/ui/tui/state/app_state.py` with typed cross-screen UI state (`active_context`, `density_mode`, notifications, selected agent id)
  - state exports in `lsm/ui/tui/state/__init__.py` for shared screen access
- Settings command-table workflow and coverage:
  - command-driven settings editing (`set`, `unset`, `delete`, `reset`, `default`, `save`) per tab
  - key/value table rendering with dirty-state markers and tab-local refresh behavior
  - new/updated tests: `tests/test_ui/tui/test_app_state.py`, `tests/test_ui/tui/test_settings_view_model.py`, `tests/test_ui/tui/test_settings_screen.py`
- Interaction-channel foundation for interactive agent handshakes:
  - new `lsm/agents/interaction.py` with `InteractionRequest`, `InteractionResponse`, and thread-safe `InteractionChannel`
  - channel capabilities: blocking request wait, non-blocking pending polling, response posting, cancellation/shutdown unblocking, and session approval caching
  - new agent config fields in `AgentConfig`: `max_concurrent` and `interaction` (`timeout_seconds`, `timeout_action`)
  - loader/serializer round-trip support for `agents.max_concurrent` and `agents.interaction`
  - configuration examples updated in `example_config.json`
  - coverage added in `tests/test_agents/test_interaction_channel.py` and expanded `tests/test_config/test_agents_config.py`
- Interaction integration across sandbox, harness, and clarification flows:
  - `ToolSandbox` now supports interactive permission confirmations through `InteractionChannel`, including session approval reuse and deterministic cancellation/shutdown unblocking
  - `AgentHarness` now accepts and propagates interaction channels, manages `WAITING_USER` status transitions while blocked on user responses, and cancels pending requests on stop
  - new built-in `ask_user` tool in `lsm/agents/tools/ask_user.py` for `"clarification"` requests, with runtime binding through harness interaction APIs
  - default tool registry now includes `ask_user`, and allowlist filtering keeps `ask_user` always available in harness/base-agent tool definitions
  - sub-agent harness/sandbox creation now inherits the parent interaction channel for consistent prompt handling across meta-agent workflows
  - coverage added in `tests/test_agents/test_sandbox_interaction.py` and `tests/test_agents/test_ask_user_tool.py`, with registry/metadata assertions updated in `tests/test_agents/test_new_tools.py` and `tests/test_agents/test_tool_risk_metadata.py`
- Multi-agent runtime manager refactor:
  - `AgentRuntimeManager` now tracks concurrent runs by `agent_id` with `AgentRunEntry` records instead of single active-agent fields
  - start flow now enforces `agents.max_concurrent`, creates per-run `InteractionChannel`, and returns generated agent ids in start output
  - runtime APIs now support targeted operations (`status/pause/resume/stop/log`) by id while keeping no-id compatibility for single-agent sessions
  - added pending-interaction APIs (`get_pending_interactions`, `respond_to_interaction`) for per-agent request/response routing
  - deterministic lifecycle handling now includes stop/shutdown cancellation + thread join behavior and bounded completed-run retention pruning
  - shell command parsing now supports optional `agent_id` arguments for status/control/log commands
  - coverage added in `tests/test_ui/shell/test_multi_agent_manager.py`
- Agents TUI interaction surface for concurrent sessions:
  - new Running Agents DataTable with per-row selection, keyboard row navigation, and periodic refresh polling
  - new Interaction Request panel with permission (`approve`, `approve_session`, `deny`) and clarification/feedback reply workflows
  - pending-interaction urgency indicator styling and panel warning-state highlighting in `lsm/ui/tui/styles/agents.tcss`
  - interaction responses route through `AgentRuntimeManager.respond_to_interaction(...)` from TUI actions/buttons
  - added `tests/test_ui/tui/test_agent_interaction.py` coverage for table selection, interaction mode switching, and response actions
- Agent interaction CLI command surface for concurrent sessions:
  - new commands: `/agent list`, `/agent interact [agent_id]`, `/agent approve <agent_id>`, `/agent deny <agent_id> [reason]`, `/agent approve-session <agent_id>`, `/agent reply <agent_id> <message>`, `/agent queue [agent_id] <message>`, `/agent select <agent_id>`
  - `/agent resume [agent_id] [message]` now supports resuming with a queued user instruction in one step
  - no-id control/log/status commands now target selected agent when available in multi-agent sessions
  - added command/parser coverage in `tests/test_ui/shell/test_agent_interaction_commands.py`
  - updated `/agent ...` completion suggestions in `lsm/ui/tui/completions.py` and `tests/test_ui/tui/test_completions.py`
- Real-time agent log streaming across harness, runtime manager, and TUI:
  - `AgentHarness` now supports optional `log_callback` invocation on every appended `AgentLogEntry`
  - `AgentRuntimeManager` now maintains bounded per-agent live log queues with dropped-entry counters and exposes `drain_log_stream(...)` for UI polling
  - running-agent startup now wires stream callbacks for both harness-produced logs and direct agent-state logs
  - agents TUI now polls selected-agent stream every 500ms and appends formatted live lines with actor prefixes (`[LLM]`, `[TOOL]`, `[AGENT]`, `[USER]`)
  - tool log lines now render as `tool(args) -> result` summaries for real-time readability
  - queue-pressure truncation is surfaced in the log panel with dropped-entry notices
  - added optional `agents.log_stream_queue_limit` config with loader/serializer and `example_config.json` coverage
  - added/updated tests in `tests/test_agents/test_harness.py`, `tests/test_ui/shell/test_multi_agent_manager.py`, `tests/test_ui/tui/test_agent_interaction.py`, and `tests/test_config/test_agents_config.py`
- Settings dirty-state and unsaved-change guards:
  - tab-level dirty indicators (`*` prefix on tab labels) with automatic sync after actions
  - `discard` and `discard tab` commands for resetting draft config
  - quit guard blocks app exit when settings have unsaved changes
  - `has_unsaved_changes` and `dirty_tab_ids` properties on `SettingsScreen`
  - `force_discard_and_leave()` for forced context switches
- Keyboard-first interaction parity across screens:
  - Agents screen keybindings: `Ctrl+Shift+R` (refresh running agents), `Ctrl+L` (show agent log), `Ctrl+I` (show agent status)
  - Remote screen keybindings: `Ctrl+Enter` (run search), `Ctrl+Shift+R` (refresh providers)
  - Updated help modal with new shortcut documentation
  - Keybinding conflict tests for all screens vs app-level bindings
- Command parsing contract tests (`tests/test_ui/test_command_parsing_contracts.py`):
  - `parse_slash_command` grammar contracts, `tokenize_command`, normalization, `parse_on_off_value`
  - `/agent` and `/memory` command grammar validation
  - Settings verb grammar and completions alignment tests
- TUI architecture documentation (`.agents/docs/architecture/development/TUI_ARCHITECTURE.md`):
  - Covers state management, worker/timer lifecycle, thread safety, error boundary, command parsing, keybinding conventions, CSS organization, and testing patterns
  - Linked from `CONTRIBUTING.md`, `TESTING.md`, and `CLAUDE.md`
- Screen presenter/controller decomposition:
  - `lsm/ui/tui/presenters/query/provider_info.py` — extracted 7 formatting functions from QueryScreen
  - `lsm/ui/tui/presenters/agents/log_formatting.py` — extracted 7 pure formatting functions from AgentsScreen
  - QueryScreen and AgentsScreen delegate to presenter functions
- Shared base screen mixin (`lsm/ui/tui/screens/base.py`):
  - `ManagedScreenMixin` provides worker/timer lifecycle delegation
  - QueryScreen, IngestScreen, AgentsScreen inherit from mixin
  - Eliminated ~200 lines of duplicated lifecycle boilerplate across 3 screens
- UI test architecture and reusable fixtures:
  - `tests/test_ui/tui/fixtures/` with `FakeStatic`, `FakeInput`, `FakeSelect`, `FakeRichLog`, `FakeButton`, `create_fake_app`
  - `tests/test_ui/tui/presenters/` for presenter-focused tests
  - `tests/test_ui/helpers/` for helper-level tests
  - Updated test organization conventions in `TESTING.md`
- TUI startup performance instrumentation and lazy background loading:
  - `StartupTimeline` dataclass in `lsm/ui/tui/app.py` records named milestones (`init_start`, `init_complete`, `mount_start`, `query_interactive`, `tui_logging_ready`, `mount_complete`, `agent_runtime_bound`, `ml_import_start`, `ml_import_complete`, `ml_model_load_start`, `ml_model_load_complete`, `background_init_complete`) with elapsed-ms tracking
  - `_schedule_background_init()` defers `_bind_agent_runtime_events()` and `_preload_ml_stack()` past first render via `call_after_refresh` in a background thread, keeping the query screen interactive immediately on launch
  - `_preload_ml_stack()` imports `sentence_transformers` and loads the configured embedding model in two phases in a background thread; the preloaded embedder is reused by `_async_init_query_context` on first query
  - `_agent_runtime_bound` flag tracks whether agent runtime binding completed for safe cleanup
  - `_trigger_agents_deferred_init()` helper triggers agents screen full init from tab activation and action switches
- Lazy ML import refactoring to eliminate eager `sentence_transformers`/`torch` import chains at startup:
  - `lsm/query/retrieval.py`: moved `from sentence_transformers import SentenceTransformer` into `_import_sentence_transformer()` helper called inside `init_embedder()`
  - `lsm/query/cost_tracking.py`: lazy-imported `build_context_block` and `prepare_local_candidates` inside function bodies
  - `lsm/ui/helpers/commands/query.py`: lazy-imported `handle_agent_command` and `handle_memory_command` inside handler functions
  - `lsm/__main__.py`: lazy-imported `run_ingest` inside `main()` dispatch branch
  - `lsm/ui/tui/__init__.py`: replaced eager `from lsm.ui.tui.app import LSMApp, run_tui` with `__getattr__` lazy re-exports
- AgentsScreen deferred initialization:
  - `on_mount()` reduced from 13 initialization calls to 3 minimal calls (`_initialize_running_controls`, `_initialize_refresh_controls`, `_focus_default_input`)
  - new `_ensure_deferred_init()` runs remaining 10 init calls + 3 timer starts on first activation, guarded by `_deferred_init_done` flag
  - deferred init triggered by: agents tab activation, runtime event messages, button presses
- TUI startup smoke tests (`tests/test_ui/tui/test_startup_smoke.py`):
  - `TestStartupSmoke`: init, compose source inspection, on_mount context, deferred agent binding, missing tabbed content resilience, lazy provider verification
  - `TestScreenImportSmoke`: all 5 screen classes importable, query screen compose source inspection, all-screen-import timing budget, retrieval module lazy ML import verification
- TUI startup performance budget tests (`tests/test_ui/tui/test_performance.py`):
  - `TestStartupTimeline`: milestone recording, unknown milestone lookup, timing growth, copy safety
  - `TestStartupPerformanceBudget`: query-interactive-under-budget enforcement (configurable via `LSM_TEST_STARTUP_BUDGET_MS`), compose import chain timing (with shared-dependency warmup via IngestScreen pre-load), `sentence_transformers` not required by screen import assertion, milestone presence, milestone ordering, background init deferral verification with ML preload mock
  - `TestAgentsDeferredInit`: minimal on_mount verification, idempotent ensure-deferred-init, runtime event trigger path
- `create_startup_mock_config()` factory in `tests/test_ui/tui/fixtures/app.py` for startup and performance test reuse
- `FakeTextArea` test fixture in `tests/test_ui/tui/fixtures/widgets.py` for query/remote log panel assertions
- Auto-focus on tab activation for Remote screen (`on_tabbed_content_tab_activated` handler)
- Auto-focus on tab activation for Agents screen (`on_tabbed_content_tab_activated` handler)
- Agents screen now displays completed agents from the current session in the Running Agents panel, marked with "[C]" prefix in the status column
- Agent screen UX refinements:
  - Unified log display format (single streaming format for both live and persisted logs)
  - Status panel auto-shows status when agent is selected in Running Agents table (removed separate Status button)
  - Follow button renamed to "Auto-Update" with corrected behavior: logs only update when Auto-Update is ON, and auto-scroll only triggers when user is already at the bottom of the log
  - Running Agents panel now displays completed agents from current session with "[C]" status indicator

### Changed

- Refactored TUI CSS from a monolithic `lsm/ui/tui/styles.tcss` into modular files under `lsm/ui/tui/styles/`:
  - `base.tcss`, `widgets.tcss`, `query.tcss`, `ingest.tcss`, `settings.tcss`, `remote.tcss`, `agents.tcss`
- Updated `LSMApp.CSS_PATH` to load split style files in deterministic order.
- Updated CSS-path expectations and CSS-loading helpers in TUI tests for the split stylesheet layout.
- Refactored settings architecture from direct in-screen mutation to state-driven controller flow:
  - `SettingsScreen` now coordinates tab activation, status, save boundary, and view-model action dispatch
  - settings rendering now uses active-tab refresh + stale-tab tracking to avoid full-screen redraw churn
- Updated settings UX from dense per-field forms to a table + command input model for faster navigation and lower render overhead.
- Updated settings tab shortcuts to function-key schema (`F2`-`F9`) and focused command input on settings load/tab switch for immediate editing.
- removed temporary compatibility fallback branches in runtime harness construction and sandbox interaction-channel wiring
- standardized Agents TUI manager call paths to finalized `agent_id`/`max_entries` signatures
- aligned shell/TUI test doubles with finalized interfaces and added callback-wiring assertions for runtime startup
- AgentsScreen `on_mount()` no longer runs full initialization; all agents-related tests now call `screen._ensure_deferred_init()` after `on_mount()` to trigger full panel setup
- `LSMApp.on_mount()` no longer calls `_bind_agent_runtime_events()` synchronously; binding is deferred via `_schedule_background_init()` to reduce startup latency
- `_schedule_background_init()` now runs agent runtime binding and ML preloading in a background thread (via `call_after_refresh`) instead of inline, preventing GIL contention from blocking the UI event loop
- `LSMApp.post_ui_message()` simplified to call `post_message()` directly (already thread-safe via `loop.call_soon_threadsafe`) instead of blocking via `call_from_thread`
- All `self.query_one(TabbedContent)` calls in `LSMApp` now use `self.query_one("#main-tabs", TabbedContent)` to avoid `TooManyMatches` ambiguity with nested `#settings-tabs`
- SettingsScreen `on_tabbed_content_tab_activated` now calls `event.stop()` to prevent nested `#settings-tabs` events from bubbling up to `LSMApp`
- Query and Remote log panels replaced `RichLog` with `TextArea(read_only=True, soft_wrap=True)` to enable text selection and copy (Ctrl+C)
- `LSMApp._write_tui_log()` updated to use `TextArea.insert()` + `scroll_end()` for log widget writes
- LLM rerank failure path changed from `_record_failure()` (ERROR level, circuit breaker increment) to `logger.warning()` with descriptive fallback message; rerank failures no longer trip the circuit breaker
- SettingsScreen `refresh_from_config()` and `on_tabbed_content_tab_activated` now guard `_focus_command_input()` behind `current_context == "settings"` check to prevent focus stealing from other screens
- `LSMApp._focus_active_screen()` and all screen `_focus_default_input()` methods now call `widget.focus()` directly instead of via `call_after_refresh()` to eliminate TabbedContent pane-switch bounce; `LSMApp.on_tabbed_content_tab_activated` now reads `event.pane.id` instead of parsing the `--content-tab-` prefixed `event.tab.id`
- Unified Agents screen log display format to a single streaming format instead of two different versions
- Renamed "Follow" button to "Auto-Update: On/Off" for clearer user understanding
- Removed Status button from Agents screen status panel; status now automatically displays when an agent is selected
- Changed Follow/Auto-Update behavior: log panel only updates when Auto-Update is ON; when turned OFF, logs stop updating until the agent is re-selected

### Fixed

- Fixed TUI log panel not displaying any log messages: Textual converts `TUILogEvent` to handler name `on_tuilog_event` (consecutive uppercase treated as one word), but the handler was named `on_tui_log_event` — all log messages were silently dropped since the handler never matched
- Fixed Settings screen flash during TUI startup where the UI briefly showed the Settings tab before switching back to Query, caused by Textual auto-focusing Input/DataTable widgets inside the Settings TabPane during compose
- Fixed `_TUILogHandler` silently swallowing delivery errors; now writes diagnostic output to stderr on failure
- Fixed query log panel text not being selectable or copyable (replaced `RichLog` with `TextArea`)
- Fixed Settings screen hotkey trap where pressing Ctrl+Q from Settings would briefly switch to Query then snap back, caused by `_focus_command_input()` firing unconditionally in `refresh_from_config()`
- Fixed LLM rerank logging `[ERROR] Provider error ... Invalid rerank response` when LLM returns malformed ranking; now logs at WARNING level with clear "falling back to local candidate ordering" message
- Fixed Remote and Agents screens not auto-focusing input when switching to their tabs via keyboard or mouse
- Fixed TUI tab-switching bounce where switching screens via keyboard shortcuts (Ctrl+Q, Ctrl+R, Ctrl+A) caused a visible triple-switch animation (e.g. Settings→Query→Settings→Query); root cause was `call_after_refresh(widget.focus)` deferring focus into the next Textual render cycle after TabbedContent had already switched panes, causing a focus-driven reverse switch; all screen focus calls are now direct `widget.focus()` calls
- Fixed `LSMApp.on_tabbed_content_tab_activated` being silently inoperative: `event.tab.id` uses a `--content-tab-{pane_id}` prefix so the previous `tab_id.replace("-tab", "")` produced strings like `--content-query` that never matched any context name; switched to `event.pane.id` which provides the pane ID directly
- Fixed `_set_active_context` processing redundant calls when both `action_switch_*` actions and the (now-working) tab activation handler both fire on the same switch; added dedup guard that returns early when `current_context` already matches the target context
- Multi-agent stop/shutdown interaction cancellation now closes per-run interaction channels, preventing stopped/completed runs from continuing to enqueue new user-interaction prompts.
- Session tool approvals selected with `approve_session` are now remembered across subsequent agent runs in the same runtime manager session.
- Agents TUI log panel now preserves manual scroll position:
  - log updates auto-scroll only when the viewer is already at the bottom
  - when scrolled up, incoming log refreshes no longer force-jump to the end
  - clicking `Log` now reloads persisted output and jumps to the bottom of the log panel
- Agents launch UX now starts runs on topic-input submit (`Enter`) and keeps control buttons in the status panel.
- Agent stop behavior now drains in-flight work more reliably before returning:
  - runtime stop waits for graceful thread completion (with bounded wait) after issuing stop
  - built-in agents (`research`, `writing`, `synthesis`, `curator`) now honor stop requests between actions and skip remaining steps
  - stop-aware completion ensures per-run log artifacts are written before stop reports success in normal cases
- Agents TUI stop control now runs stop requests on a background worker to prevent UI hangs during graceful-stop waits.
- Agents screen panel order updated so launch appears first and status/control appears immediately after launch.
- Fixed Agents TUI multiple log display formats: unified log display to use streaming format for both live and persisted logs
- Fixed Agents TUI status not showing on agent selection: removed status button, now auto-displays status when agent is selected in Running Agents table
- Fixed Agents TUI follow button behavior: renamed to "Auto-Update", now only updates log panel when enabled and user is at bottom; when disabled, logs stop updating until agent is reselected
- Fixed Agents TUI showing completed agents: Running Agents table now displays completed agents from current session with "[C]" status indicator
- Config path resolution is now consistent for vector persistence and ingest artifacts:
  - relative `global.global_folder` now resolves from the config file directory
  - `vectordb.persist_dir` is the single source of truth; `ingest.persist_dir` is ignored on load and no longer serialized
  - relative `ingest.manifest` now resolves under `global.global_folder` (falling back to config directory when `global_folder` is unset)
  - ingest stats cache now reads and writes under `vectordb.persist_dir`

## 0.5.0 - 2026-02-13

### Added

- Agent memory storage backend foundation in `lsm/agents/memory/`:
  - memory datamodels (`Memory`, `MemoryCandidate`)
  - backend abstraction (`BaseMemoryStore`)
  - SQLite and PostgreSQL memory store implementations
  - backend factory with `auto` selection
  - migration helpers for SQLite <-> PostgreSQL memory stores
- New `MemoryConfig` in `lsm/config/models/agents.py` with:
  - `storage_backend` (`auto` | `sqlite` | `postgresql`)
  - `sqlite_path`, `postgres_connection_string`, `postgres_table_prefix`
  - TTL cap settings (`ttl_project_fact_days`, `ttl_task_state_days`, `ttl_cache_hours`)
- Memory store coverage tests in `tests/test_agents/test_memory_store.py`:
  - CRUD, promotion/rejection lifecycle, expiry cleanup
  - TTL cap enforcement and backend auto-selection
  - store-to-store migration behavior
- Memory API module `lsm/agents/memory/api.py` with:
  - `memory_put_candidate()`, `memory_promote()`, `memory_expire()`
  - `memory_search()` ranking by recency + pin weighting
- Memory standing-context builder in `lsm/agents/memory/context_builder.py` for pre-LLM prompt injection.
- Memory API/context integration tests:
  - `tests/test_agents/test_memory_api.py`
  - `tests/test_agents/test_memory_context_builder.py`
- Memory agent tools:
  - `memory_put` (`writes_workspace`, `requires_permission=True`) for proposing candidates and updating existing memories by `memory_id`
  - `memory_remove` (`writes_workspace`, `requires_permission=True`) for deleting memory records
  - `memory_search` (`read_only`) for querying promoted memories
- Memory tool coverage in `tests/test_agents/test_memory_tools.py`.
- Per-run agent summary artifact in `lsm/agents/harness.py`:
  - emits `run_summary.json` for every harness run
  - captures tool usage, permission approvals/denials, outcome, duration, token usage, and constraints
- Curator memory-distillation mode in `lsm/agents/curator.py`:
  - supports `--mode memory` topic flag (and `agent_overrides["mode"]`)
  - scans recent `run_summary.json` artifacts
  - emits `memory_candidates.md` and `memory_candidates.json`
- Curator memory mode and run-summary coverage tests:
  - `tests/test_agents/test_curator_memory_mode.py`
  - `tests/test_agents/test_harness.py`
  - `tests/test_agents/test_harness_allowlist.py`
- Memory UI/command surfaces:
  - `/memory` command handler in `lsm/ui/shell/commands/agents.py` with candidate listing, promote/reject, and TTL editing
  - Memory command routing in `lsm/ui/tui/screens/query.py`
  - Memory candidates panel in `lsm/ui/tui/screens/agents.py` with refresh/approve/reject/edit TTL actions
  - Query command completion support for `/memory` in `lsm/ui/tui/completions.py`
- Memory UI test coverage:
  - `tests/test_ui/shell/test_memory_commands.py`
  - `tests/test_ui/tui/test_memory_screen.py`
- Live PostgreSQL memory backend coverage in `tests/test_agents/test_live_memory_store_postgresql.py`:
  - CRUD/promotion/search/delete on `PostgreSQLMemoryStore`
  - reject/expire/TTL-cap behavior
  - PostgreSQL -> SQLite memory migration validation
- Base provider message transport interface in `BaseLLMProvider` with `_send_message(...)` and `_send_streaming_message(...)`.
- Shared fallback answer helper on `BaseLLMProvider` to replace per-provider `_fallback_answer()` duplication.
- Shared provider JSON schema constants in `lsm/providers/helpers.py`:
  - `RERANK_JSON_SCHEMA`
  - `TAGS_JSON_SCHEMA`
- Test infrastructure for tiered execution (`smoke`, `integration`, `live`, `live_llm`, `live_remote`, `live_vectordb`, `docker`, `performance`) with default exclusion of `live` and `docker`.
- `tests/testing_config.py` + `tests/.env.test.example` support for `LSM_TEST_*` runtime configuration, including `LSM_TEST_POSTGRES_CONNECTION_STRING`.
- Tier-aware pytest fixtures for real providers and services in `tests/conftest.py`:
  - real embedder and ChromaDB fixtures
  - live LLM provider fixtures (OpenAI, Anthropic, Gemini, Local/Ollama)
  - PostgreSQL preflight fixture with connectivity, `CREATE` privilege, and `pgvector` checks
- Comprehensive synthetic corpus in `tests/fixtures/synthetic_data/` for long-form, edge-case, nested-tag, duplicate, and config-driven tests.
- Real/live Phase 2 suites:
  - real embedding tests (`tests/test_ingest/test_real_embeddings.py`)
  - real ChromaDB tests (`tests/test_vectordb/test_real_chromadb.py`)
  - live LLM provider tests (`tests/test_providers/test_live_*.py`)
  - live remote provider tests across the built-in remote providers (`tests/test_providers/remote/test_live_*.py`)
  - end-to-end integration pipeline tests (`tests/test_integration/test_full_pipeline.py`)
  - live PostgreSQL vector DB tests (`tests/test_vectordb/test_live_postgresql.py`)
  - live ChromaDB->PostgreSQL migration tests (`tests/test_vectordb/test_live_migration_chromadb_to_postgres.py`)
- Full live pipeline coverage using PostgreSQL/pgvector as the backing vector store (`@pytest.mark.live` + `@pytest.mark.live_vectordb` path in `tests/test_integration/test_full_pipeline.py`).
- Mock-cleanup guardrail test (`tests/test_infrastructure/test_mock_audit.py`) to prevent reintroduction of removed legacy shared mock fixtures in core integration suites.
- Scheduler configuration model in `lsm/config/models/agents.py`:
  - added `ScheduleConfig` (`agent_name`, `params`, `interval`, `enabled`, `concurrency_policy`, `confirmation_mode`)
  - added `AgentConfig.schedules` for typed schedule entries
- Scheduler config loader/serializer support in `lsm/config/loader.py`:
  - parses `agents.schedules` from raw config
  - round-trips schedules through `config_to_raw(...)`
- Scheduler config documentation/examples:
  - `example_config.json` includes `agents.schedules` example
- `.agents/docs/architecture/development/AGENTS.md` agent config reference now documents schedules
- Scheduler config tests in `tests/test_config/test_schedule_config.py` covering model validation, normalization, loader parsing, and serialization.
- `AgentScheduler` engine in `lsm/agents/scheduler.py`:
  - persistent schedule runtime state at `<agents_folder>/schedules.json`
  - threaded tick loop with due-schedule execution and graceful stop handling
  - overlap policies: `skip`, `queue`, `cancel`
  - schedule status metadata (`last_run_at`, `next_run_at`, `last_status`, `last_error`)
  - interval parsing for aliases (`hourly`, `daily`, `weekly`), seconds (`<n>s`), and cron syntax
- Scheduled-run safety policy enforcement in scheduler:
  - read-only tools only by default
  - network disabled by default
  - explicit schedule `params` opt-in for writes/network/exec
  - network/exec runs force sandbox `execution_mode="prefer_docker"`
  - optional schedule `params.force_docker=true` to require Docker runner for all tool risks
- Scheduler test coverage in `tests/test_agents/test_scheduler.py`:
  - persistence/reload behavior
  - no-overlap guarantees and concurrency-policy behavior
  - safe-default and explicit-opt-in tool/sandbox behavior
  - due-tick execution logic
- Scheduler shell command management in `lsm/ui/shell/commands/agents.py`:
  - `/agent schedule add|list|enable|disable|remove|status`
  - schedule JSON params parsing via `--params '{...}'`
  - `--concurrency_policy` and `--confirmation_mode` support on `schedule add`
  - schedule edits persist to `config.json`/`config.yaml` when `config_path` is available
  - runtime manager wiring to create/rebuild scheduler state from command changes
- Agents TUI schedule panel in `lsm/ui/tui/screens/agents.py`:
  - table-based schedule list with refresh, add/remove, and enable/disable controls
  - runtime status output from scheduler state
- Scheduler UI test coverage:
  - `tests/test_ui/shell/test_schedule_commands.py`
  - `tests/test_ui/tui/test_schedule_screen.py`
- Sandbox runner policy now supports `sandbox.force_docker`:
  - all tool risks (`read_only`, `writes_workspace`, `network`, `exec`) require Docker when enabled
  - execution is blocked with `PermissionError` when Docker is unavailable
- Meta-agent core orchestration foundation:
  - new `TaskGraph` + `AgentTask` models in `lsm/agents/task_graph.py` with dependency validation, topological sorting, ready-task selection, and status transitions
  - new `MetaAgent` in `lsm/agents/meta.py` that converts user goals into deterministic task graphs and dependency-ordered execution plans (planning-only core mode)
  - `meta` agent registration in `lsm/agents/factory.py` and framework exports in `lsm/agents/__init__.py`
  - coverage in `tests/test_agents/test_task_graph.py` and `tests/test_agents/test_meta_agent.py`
- Meta-agent system tools:
  - new `spawn_agent` tool in `lsm/agents/tools/spawn_agent.py` for launching sub-agent runs
  - new `await_agent` tool in `lsm/agents/tools/await_agent.py` for blocking until a spawned run completes
  - new `collect_artifacts` tool in `lsm/agents/tools/collect_artifacts.py` for retrieving spawned-run artifacts by glob pattern
  - default tool registry registration and exports for all three tools
- Meta-agent phase 6.2 coverage:
  - `tests/test_agents/test_meta_tools.py`
  - `tests/test_agents/test_sandbox_monotone.py`
- Meta-agent phase 6.3 shared-workspace execution:
  - `MetaAgent` now initializes per-run workspace roots with:
    - `<agents_folder>/meta_<timestamp>/workspace/`
    - `<agents_folder>/meta_<timestamp>/sub_agents/<agent_name>_<NNN>/`
    - `final_result.md` and `meta_log.md`
  - sub-agent sandboxes are scoped so shared workspace is readable and each sub-agent workspace is the only writable path
  - final synthesis writes `final_result.md` via provider when available, with deterministic fallback output
  - meta log output includes task trace plus structured agent log entries
- Meta-agent phase 6.3 coverage:
  - `tests/test_agents/test_meta_workspace.py`
  - `tests/test_agents/test_meta_synthesis.py`
- Meta-agent phase 6.4 UI/CLI:
  - shell commands in `lsm/ui/shell/commands/agents.py`:
    - `/agent meta start <goal>`
    - `/agent meta status`
    - `/agent meta log`
  - `AgentRuntimeManager` meta snapshot/status/log helpers for task graph, sub-agent run state, and persisted meta log access
  - Agents TUI meta panel in `lsm/ui/tui/screens/agents.py`:
    - task-graph progress table
    - sub-agent run status/workspace table
    - artifact list with `final_result.md` and `meta_log.md` paths
  - query command autocomplete support for `/agent meta ...` in `lsm/ui/tui/completions.py`
- Meta-agent phase 6.4 coverage:
  - `tests/test_ui/shell/test_meta_commands.py`
  - `tests/test_ui/tui/test_meta_screen.py`
  - updated `tests/test_ui/tui/test_completions.py`

### Changed

- Documentation refresh for v0.5.0 across:
  - `docs/README.md`
- `.agents/docs/architecture/development/AGENTS.md`
- `.agents/docs/architecture/development/PROVIDERS.md`
- `.agents/docs/architecture/development/TESTING.md`
- Agent config loader/serializer now round-trips `agents.memory` settings via `build_memory_config(...)` and `config_to_raw(...)`.
- Relative `agents.memory.sqlite_path` values now resolve under resolved `agents_folder` (which itself resolves under `global_folder` when configured).
- `BaseMemoryStore` now supports `mark_used(memory_ids, used_at=...)` to update `last_used_at` for injected memories.
- `AgentHarness` now injects a separate standing-memory context block before each LLM turn when memory context is available.
- `AgentHarness` action-loop flow now records permission decisions and tool execution telemetry for run-summary generation.
- Query/TUI command handling now supports `/memory` command workflows in addition to `/agent`.
- `create_default_tool_registry()` now registers memory tools when memory is enabled and a memory store is supplied; shell agent startup initializes and injects the configured memory backend.
- `AgentHarness` now supports spawned sub-agent execution with runtime tracking (`spawn_sub_agent`, `await_sub_agent`, `collect_sub_agent_artifacts`) and per-sub-agent scoped work areas.
- `ToolSandbox` global-subset validation now enforces monotonicity across:
  - `allow_url_access`
  - `force_docker`
  - `execution_mode`
  - `require_user_permission` and `require_permission_by_risk` gates
  - runtime limits (`timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`)
- Moved shared LLM business logic into `BaseLLMProvider` concrete methods:
  - `rerank(...)`
  - `synthesize(...)`
  - `stream_synthesize(...)`
  - `generate_tags(...)`
- Refactored OpenAI, Azure OpenAI, Anthropic, Gemini, and Local providers to keep provider-specific transport/config logic only.
- Removed duplicated `rerank/synthesize/stream_synthesize/generate_tags` implementations from provider classes in favor of base implementations.
- Verified provider factory and exports remain correct after refactor (`lsm/providers/factory.py`, `lsm/providers/__init__.py`).
- Migrated core ingest/query integration tests away from `mocker`/`unittest.mock` toward lightweight fake implementations plus `monkeypatch`.
- Removed legacy shared mock fixtures from `tests/conftest.py`:
  - `mock_openai_client`
  - `mock_embedder`
  - `mock_chroma_collection`
  - `mock_vectordb_provider`
  - `progress_callback_mock`
- Updated live Anthropic model selection in tests to `claude-sonnet-4-5`.

### Fixed

- Fixed PostgreSQL vector similarity SQL parameter typing by casting query parameters to `vector` (`%s::vector`) in `lsm/vectordb/postgresql.py`, resolving `vector <=> numeric[]` operator errors in live runs.
- Fixed ChromaDB->PostgreSQL migration handling of numpy embeddings in `lsm/vectordb/migrations/chromadb_to_postgres.py` by normalizing embeddings without ambiguous truth-value evaluation.

## 0.4.0 - 2026-02-08

### Added

- Config restructuring with a dedicated top-level `global` object and zero flat top-level fields.
- LLM providers/services registry pattern (`llms.providers` + `llms.services`) with service resolution and fallback to `default`.
- Per-provider LLM pricing tables and cost estimation integrated into query cost tracking.
- Structure-aware chunking in `lsm/ingest/structure_chunking.py` that respects headings, paragraphs, and sentence boundaries.
- Heading detection for Markdown (`#`-style) and bold-only lines (common in PDF extractions).
- Page number tracking for PDF and DOCX documents via `PageSegment` dataclass.
- DOCX page break detection using `<w:lastRenderedPageBreak/>` and `<w:br w:type="page"/>` XML elements.
- `chunking_strategy` config option (`"structure"` or `"fixed"`) on `IngestConfig`.
- Chunk metadata now includes `heading`, `paragraph_index`, and `page_number` fields when using structure chunking.
- Language detection module (`lsm/ingest/language.py`) using `langdetect` for automatic document language identification (ISO 639-1 codes).
- `enable_language_detection` config option on `IngestConfig` (default `False`). Detected language stored in chunk metadata as `"language"`.
- LLM-based machine translation module (`lsm/ingest/translation.py`) for cross-language search on multilingual corpora.
- `enable_translation` and `translation_target` config options on `IngestConfig`. Uses `"translation"` LLM service from `llms.services`.
- `WELL_KNOWN_EMBED_MODELS` dictionary in `constants.py` mapping 30+ embedding models to their output dimensions.
- `embedding_dimension` field on `GlobalConfig` with auto-detection from well-known models. Pipeline validates actual model dimension matches config at startup.
- `RootConfig` dataclass in `lsm/config/models/ingest.py` supporting per-root `tags` and `content_type`.
- `IngestConfig.roots` now accepts strings, Path objects, dicts with `path`/`tags`/`content_type`, or `RootConfig` instances (all normalized to `List[RootConfig]`).
- `root_paths` property on `IngestConfig` for convenient `List[Path]` access.
- `.lsm_tags.json` subfolder tag support via `collect_folder_tags()` in `lsm/ingest/fs.py`.
- Root tags, content type, and folder tags propagated to chunk metadata as `root_tags`, `content_type`, and `folder_tags`.
- `iter_files()` now yields `(Path, RootConfig)` tuples to track file-to-root mapping.
- 30 new tests in `tests/test_ingest/test_root_config.py` covering RootConfig, config loading, folder tag discovery, and iter_files changes.
- Partial ingest support via `max_files` and `max_seconds` limits on `IngestConfig`. Pipeline cleanly flushes queues and saves manifest when limits are reached.
- Stats caching via `StatsCache` class in `lsm/ingest/stats_cache.py`. Cache stored at `<persist_dir>/stats_cache.json` with staleness detection (count mismatch or age expiry). Automatically invalidated after ingest.
- Enhanced MuPDF PDF repair with multi-stage strategy: direct open, garbage-collection rebuild (`garbage=4, deflate=True, clean=True`), and plain stream fallback. Expanded repairable error markers to include `"trailer"`, `"startxref"`, `"corrupt"`, and `"malformed"`.
- Chunk version control via `enable_versioning` on `IngestConfig`. Old chunks marked `is_current=False` instead of deleted. Version number tracked in chunk metadata and manifest. Query retrieval filters to `is_current=True` when versioning is active.
- `where_filter` parameter on `retrieve_candidates()` for metadata-level filtering at query time.
- Full PostgreSQL + pgvector support as an alternative vector database backend via `VectorDBConfig(provider="postgresql")`.
- `VectorDBGetResult` dataclass in `lsm/vectordb/base.py` for typed non-similarity retrieval results (ids, documents, metadatas, embeddings).
- `get()` abstract method on `BaseVectorDBProvider` supporting retrieval by IDs, filters, with pagination (`limit`/`offset`) and field selection (`include`).
- `update_metadatas()` and `delete_all()` abstract methods on `BaseVectorDBProvider`, implemented on both ChromaDB and PostgreSQL providers.
- `PostgreSQLProvider` with full implementation: `add_chunks()`, `query()`, `get()`, `update_metadatas()`, `delete_by_filter()`, `delete_all()`, `count()`, `get_stats()`, and `_normalize_filters()` for JSONB containment queries.
- ChromaDB-to-PostgreSQL migration tool in `lsm/vectordb/migrations/chromadb_to_postgres.py` with batched reads/writes and progress callback.
- `migrate-vectordb` CLI subcommand and `/migrate` TUI command for running migrations.
- PostgreSQL connection variables in `.env.example` (`LSM_POSTGRES_CONNECTION_STRING`, `LSM_POSTGRES_TABLE`).
- PostgreSQL vectordb example in `example_config.json`.
- Query result cache in `lsm/query/cache.py` with TTL expiration + LRU eviction, integrated into query execution behind `query.enable_query_cache`.
- Query chat modes (`query.chat_mode = single|chat`) with `/mode chat` and `/mode single` switching in TUI query commands.
- Global chat transcript settings via `ChatsConfig` (`enabled`, `dir`, `auto_save`, `format`) with transcript auto-save support.
- Chat conversation/session tracking in query state, including provider response/session ID chaining for follow-up turns.
- Provider-side LLM cache/session reuse support across OpenAI, Azure OpenAI, Anthropic, Gemini, and Local providers (where applicable by provider API).
- TUI live mode toggle for provider cache reuse: `/mode set llm_cache on|off`.
- Query metadata prefiltering improvements (`lsm/query/prefilter.py`, `lsm/query/planning.py`) using metadata inventory + deterministic author/year/title extraction.
- Prefiltering now supports all tag fields together: `ai_tags`, `user_tags`, `root_tags`, `folder_tags`, plus `content_type`.
- Context anchor controls in query TUI (`/context`, `/context doc ...`, `/context chunk ...`, `/context clear`) with anchor-first context prioritization.
- Added/updated query tests for metadata prefilter behavior and anchor prioritization (`tests/test_query/test_prefilter.py`, `tests/test_query/test_planning.py`).
- Natural language query decomposition in `lsm/query/decomposition.py` with `QueryFields` (`author`, `keywords`, `title`, `date_range`, `doi`, `raw_query`).
- Added deterministic field extraction (`extract_fields_deterministic`) and AI-assisted extraction (`extract_fields_ai`) with structured JSON parsing and deterministic fallback.
- Added decomposition dispatcher `decompose_query(method="deterministic"|"ai")` and test coverage in `tests/test_query/test_decomposition.py`.
- Added `llms.services.decomposition` support so query decomposition uses a dedicated configurable model/provider.
- Mode-level chat save overrides via `modes[].chats` (`auto_save`, `dir`) for per-mode transcript behavior.
- Dict-based remote provider protocol in `lsm/remote/base.py` and all built-in providers (`search_structured`, `get_input_fields`, `get_output_fields`, `get_description`) with normalized output fields (`url`, `title`, `description`, `doi`, `authors`, `year`, `score`, `metadata`).
- Remote provider structured protocol test coverage in `tests/test_providers/remote/test_structured_protocol.py`.
- Remote result disk caching via `lsm/remote/storage.py` with `save_results()` and `load_cached_results()`, integrated into query remote fetch flow.
- Remote provider config keys `cache_results` and `cache_ttl` for per-provider cache control.
- Remote provider chaining with `remote_provider_chains` config, `ChainLink` mapping (`output:input`), and `RemoteProviderChain` execution in `lsm/remote/chain.py`.
- Restructured TUI settings screen to config-aligned sub-tabs: `Global`, `Ingest`, `Query`, `LLM`, `Vector DB`, `Modes`, `Remote`, and `Chats/Notes`, with section-level Save/Reset controls.
- Live settings updates in TUI now write directly to `app.config` for edited fields, with validation/status feedback and config reload-on-reset.
- Agent framework foundation in new `lsm/agents/` package with `AgentStatus`, `AgentState`, `BaseAgent`, and core runtime models (`AgentLogEntry`, `ToolResponse`, `AgentContext`).
- New optional `agents` config block with `AgentConfig` and `SandboxConfig`, including loader/serializer support for `agents_folder`, token/iteration limits, context strategy, sandbox permissions, and per-agent overrides.
- Agent tool system in `lsm/agents/tools/` with `BaseTool`, `ToolRegistry`, `ToolSandbox`, and default tools for file IO, URL loading, local embedding queries, LLM prompting, remote provider queries, and remote chain execution.
- Agent runtime engine in `lsm/agents/harness.py` with JSON action loop, tool execution, context-window strategies (`compact` and `fresh`), background thread execution, pause/resume/stop controls, budget/iteration guards, and persisted per-run state files under `agents_folder`.
- Agent log helpers in `lsm/agents/log_formatter.py` for formatting, saving, and loading structured `AgentLogEntry` traces.
- Research agent in `lsm/agents/research.py` with LLM-driven topic decomposition, per-subtopic tool selection, iterative evidence synthesis, outline review loops, and markdown outline persistence.
- Agent registry/factory in `lsm/agents/factory.py` with built-in `research` registration and `create_agent()` entrypoint for extensible agent creation.
- Agent UI integration with new `lsm/ui/tui/screens/agents.py` tab (launcher, status, pause/resume/stop, log view) wired into `LSMApp`.
- Added shell-level agent command handlers in `lsm/ui/shell/commands/agents.py` supporting `/agent start|status|pause|resume|stop|log`, and query screen routing for `/agent` commands.
- Tool risk metadata on `BaseTool`: `risk_level` (`read_only` | `writes_workspace` | `network` | `exec`), `preferred_runner` (`local` | `docker`), `needs_network`. All 15 built-in tools tagged with appropriate risk levels. `ToolRegistry` gains `list_by_risk()` and `list_network_tools()` query methods.
- Sandbox security hardening in `lsm/agents/tools/sandbox.py`:
  - `_canonicalize_path()` with null byte rejection, Windows UNC path and alternate data stream blocking, symlink escape detection, and post-normalization `..` rejection.
  - Environment scrubbing in `lsm/agents/tools/env_scrubber.py` — minimal env (PATH, HOME, TEMP, LANG) with `*_API_KEY`, `*_SECRET*`, `*_TOKEN*`, `*_PASSWORD*` exclusion.
  - Log redaction in `lsm/agents/log_redactor.py` — masks `sk-*`, `key_*`, base64 blobs, and secret assignment patterns in all agent logs and tool output.
  - Interactive permission gates in `lsm/agents/permission_gate.py` with `PermissionGate` class and `PermissionDecision` dataclass. Precedence: per-tool override > per-risk policy > tool default > allow.
- `SandboxConfig` extensions: `execution_mode` (`local_only` | `prefer_docker`), `require_permission_by_risk` dict, `limits` (timeout, max stdout, max file write), `docker` settings (image, network, CPU/memory limits, read-only root).
- Runner abstraction in `lsm/agents/tools/runner.py` with `ToolExecutionResult` dataclass, `BaseRunner` ABC, and `LocalRunner` implementation with timeout enforcement, output truncation, and write-size limits.
- Docker sandbox foundation: `DockerRunner` in `lsm/agents/tools/docker_runner.py` using subprocess `docker run` with workspace mounts, resource limits, and JSON payload protocol. `Dockerfile.sandbox` with Python 3.12-slim, non-root user, and read-only root. Entrypoint in `lsm/agents/tools/_docker_entrypoint.py`. Runner selection policy routes `read_only`/`writes_workspace` to LocalRunner and `network`/`exec` to DockerRunner when available.
- Six new agent tools:
  - `extract_snippets` — semantic snippet retrieval scoped to specific file paths.
  - `file_metadata` — file size, mtime, and extension for path lists.
  - `hash_file` — SHA256 hash computation for duplicate detection.
  - `similarity_search` — cosine similarity pairs for chunk IDs or source paths.
  - `source_map` — evidence aggregation by source path with count and top snippets.
  - `append_file` — append content to existing files (`writes_workspace`, requires permission).
- Agent framework flexibility enhancements in `lsm/agents/base.py`: extracted `_log()`, `_parse_json()`, `_consume_tokens()`, `_budget_exhausted()` helpers from `ResearchAgent` to `BaseAgent`. Added `_format_tool_definitions_for_prompt()` and `_parse_tool_selection()` for standardized tool formatting and LLM response parsing.
- Tool allowlist on `AgentHarness`: `tool_allowlist` parameter filters tool definitions sent to the LLM and blocks execution of unlisted tools.
- Automatic per-run workspace creation: harness creates `<agents_folder>/<agent_name>_<timestamp>/workspace/` and passes path in agent context.
- Writing agent in `lsm/agents/writing.py` with grounding retrieval, outline building, prose drafting, self-review, and `deliverable.md` persistence. Tool allowlist: `query_embeddings`, `read_file`, `read_folder`, `write_file`, `extract_snippets`, `source_map`.
- Synthesis agent in `lsm/agents/synthesis.py` with scope selection, candidate gathering, format-driven synthesis (bullets/outline/narrative/QA), tightening, coverage check, and `synthesis.md` + `source_map.md` persistence. Tool allowlist: `read_folder`, `query_embeddings`, `read_file`, `write_file`, `extract_snippets`, `source_map`.
- Curator agent in `lsm/agents/curator.py` with file inventory, metadata collection, SHA256 exact-duplicate detection, embedding-based near-duplicate detection, staleness/quality heuristics, LLM-driven recommendations, and `curation_report.md` persistence. Tool allowlist: `read_folder`, `file_metadata`, `hash_file`, `query_embeddings`, `similarity_search`, `write_file`.
- Agent factory updated to register `writing`, `synthesis`, and `curator` agents alongside `research`.
- Comprehensive adversarial security test suite (50 tests across 7 files) covering STRIDE threat categories:
  - T1: Arbitrary file access — path traversal, null bytes, UNC paths, ADS, symlink escape (`test_security_paths.py`, 12 tests)
  - T2+T3: Command execution and privilege escalation — registry, schema, permission gates, risk policies (`test_security_permissions.py`, 8 tests)
  - T4: Network abuse — tool blocking when `allow_url_access=False` (`test_security_network.py`, 5 tests)
  - T5: Resource exhaustion — iteration caps, token budgets, output truncation (`test_security_resources.py`, 5 tests)
  - T6: Data integrity — write boundaries, safe directory creation, artifact tracking (`test_security_integrity.py`, 4 tests)
  - T7: Prompt injection — non-JSON responses, malformed actions, embedded JSON isolation (`test_security_injection.py`, 7 tests)
  - T8: Secret leakage — env scrubbing, log redaction, base64 masking, harness integration (`test_security_secrets.py`, 9 tests)
- Security documentation in `.agents/docs/architecture/development/SECURITY.md` with threat model, attack surface inventory, STRIDE coverage matrix, adversarial testing methodology, extension guide, and permission gate reference.

### Changed

- `parse_pdf()` and `parse_docx()` now return 3-tuples `(text, metadata, page_segments)` to preserve page boundary information.
- `parse_file()` updated to return 3-tuples consistently across all formats (page_segments is `None` for non-paginated formats).
- Pipeline writer thread now writes `heading`, `paragraph_index`, and `page_number` into vector DB chunk metadata.
- Default chunking strategy is `"structure"`; legacy fixed-size chunking available via `"fixed"`.
- All consumer modules (`stats.py`, `tagging.py`, `api.py`, `planning.py`, `retrieval.py`, `pipeline.py`, TUI screens/commands) now use `BaseVectorDBProvider` interface exclusively — no raw ChromaDB imports outside `lsm/vectordb/chromadb.py`.
- `init_collection()` removed from `retrieval.py` — consumers use `create_vectordb_provider()` factory.
- `get_by_filter()` removed from `BaseVectorDBProvider` — replaced by `get(filters=...)`.
- `require_chroma_collection()` utility removed — no longer needed with provider abstraction.
- `lsm/vectordb/utils.py` deleted entirely.
- Filter format normalized: simple `{"key": "value"}` instead of `{"key": {"$eq": "value"}}` at the provider interface level.
- `query.enable_llm_server_cache` default is now `true`.
- Removed `llm_prompt_cache_retention` query config option; provider backends control retention policy.
- Query planning now resolves `llms.services.decomposition` and passes that model config into metadata prefilter/decomposition flow.
- Chat auto-save now applies mode-level overrides before saving transcripts (global defaults can be overridden per mode).

## 0.3.2

### Added

- Clean ingest API in `lsm/ingest/api.py` with typed results for ingest, stats, info, and wipe operations.
- Progress callback support across ingest and query flows, including TUI progress integration.
- Shared LLM provider helpers in `lsm/providers/helpers.py` to centralize prompts, parsing, and fallback behavior.
- Global path management in `lsm/paths.py` with default user folder support for chats and notes.
- Expanded test coverage with new vector DB, ingest API, config, logging, and path test suites.
- New integration test fixtures and suites for ingest pipeline and query progress callbacks.

### Changed

- Refactored `lsm.ingest` toward a cleaner architecture that separates business logic from UI command handling.
- Consolidated duplicated provider logic across OpenAI, Anthropic, Gemini, Azure OpenAI, and Local providers.
- Removed legacy configuration fallbacks and typo-tolerant compatibility paths from config loading.
- Simplified remote provider activation semantics by mode-driven selection.
- Improved lazy loading for package/provider/vector DB components to reduce startup overhead.

### Fixed

- Improved ingest and query fault tolerance with better error handling and partial-result resilience.
- Added graceful handling for provider failures and remote timeout scenarios.
- Removed deprecated/legacy ingest code paths that conflicted with current architecture guidance.

## 0.3.1

### Added

- **TUI (Textual User Interface)** - Rich terminal interface with:
  - Tabbed navigation (Query, Ingest, Settings)
  - ResultsPanel widget with expandable citations
  - CommandInput widget with history and Tab autocomplete
  - StatusBar widget showing mode, chunks, cost, provider status
  - Keyboard shortcuts (Ctrl+B to build, Ctrl+E to expand, etc.)
  - Help modal (F1) with command reference
- Documentation expansion across user guide, architecture, API, and dev guides.
- Refinements to configuration reference.
- Added Anthropic, Gemini, Local (Ollama), and Azure OpenAI providers.
- Added provider health tracking and `/provider-status` REPL command.
- LLM configuration now uses an ordered `llms` list with per-feature selection.

### Changed

- Module restructuring for GUI preparation:
  - CLI/REPL code moved to `lsm/gui/shell/`
  - Remote providers moved to `lsm/remote/`
  - Split large REPL files into modular components

## 0.2.0

### Added

- Unified interactive shell with ingest and query contexts.
- Query modes (`grounded`, `insight`, `hybrid`) and mode switching in REPL.
- Notes system with `/note` and Markdown output.
- Remote provider framework with Brave Search integration.
- AI tagging for chunks in ingest REPL.

### Changed

- LLM configuration consolidated under `llm` with per-feature overrides.
- LLM configuration now supports ordered multi-provider selection via `llms`.
- Query pipeline now supports hybrid reranking and relevance gating.

### Fixed

- Incremental ingest reliability via manifest hash checks.

### Breaking Changes

- None known.

### Migration Notes

- Legacy `openai` and single-provider `llm` sections are removed; migrate to
  the ordered `llms` list schema.

## 0.1.0

### Added

- Initial local-first ingest pipeline with ChromaDB storage.
- Query pipeline with semantic retrieval and citations.
- Basic CLI entrypoints for ingest and query.
