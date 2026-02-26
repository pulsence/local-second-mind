# Phase 7: query_remote Tool Redesign

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1

Redesigns `QueryRemoteTool` from a single tool that accepts a `provider` name argument to a
factory-based pattern where each configured remote source gets its own named tool instance.
This gives the LLM a clear, typed interface per source and removes the need for the LLM to
know internal provider names.

**Current design**: Single `QueryRemoteTool` with `name = "query_remote"`. LLM passes
`provider` name as an argument. Tool looks up the provider in config at runtime.

**Proposed design**: Each configured remote source becomes its own tool instance:
- `QueryRemoteTool` is parameterized per source at construction time
- Instance `name` property returns `f"query_{provider_cfg.name}"` (e.g., `query_brave_search`)
- Instance `description` is derived from the provider's configured description or type
- Instance `input_schema` reflects the provider's expected input structure
- `create_default_tool_registry()` instantiates one `QueryRemoteTool` per configured remote source
- Agents declare which remote sources they want via a new class attribute (e.g.,
  `remote_source_allowlist: Optional[set[str]] = None`); `None` means all available sources
- The `AgentHarness` filters per-source tool instances against the agent's `remote_source_allowlist`
  when building the tool list passed to the LLM

---

## 7.1: QueryRemoteTool — Per-Source Instance Design

**Description**: Redesign `QueryRemoteTool` to be parameterized by a single `RemoteProviderConfig`.
Each instance represents exactly one remote source.

**Tasks**:
- Refactor `QueryRemoteTool.__init__` to accept `RemoteProviderConfig` and `LSMConfig`:
  - `self._provider_cfg = provider_cfg`
  - `self.config = config`
- Change `name` to a property: `return f"query_{self._provider_cfg.name}"`
- Change `description` to reflect the specific source (e.g., `f"Query {self._provider_cfg.name} ({self._provider_cfg.type}) for structured information."`)
- Update `input_schema` to remove the `provider` parameter (no longer needed; the tool IS
  the provider); keep `input` and `max_results`
- Rewrite `execute()` to use `self._provider_cfg` directly — no `_find_provider()` lookup needed
- Delete `_find_provider()` helper
- Preserve `_provider_config_to_dict()` as a private helper (or inline it)
- Run relevant tests: `pytest tests/agents/tools/test_query_remote.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/tools/query_remote.py` — refactor `QueryRemoteTool`
- `tests/agents/tools/test_query_remote.py` — update:
  - Test: `QueryRemoteTool(provider_cfg, config).name` returns `f"query_{provider_cfg.name}"`
  - Test: `execute()` does not accept or use a `provider` argument
  - Test: `execute()` calls the correct provider directly
  - Test: two `QueryRemoteTool` instances for different configs have different names

**Success criteria**: `QueryRemoteTool` instances are provider-specific. Tool name reflects
the provider name. No `provider` parameter in `input_schema` or `execute()`.

---

## 7.2: Tool Registry — Per-Source Tool Instantiation

**Description**: Update `create_default_tool_registry()` to instantiate one `QueryRemoteTool`
per configured remote source. Add `remote_source_allowlist` to `BaseAgent`. Update
`AgentHarness` to filter per-source tools against the allowlist.

**Tasks**:
- In `create_default_tool_registry()`:
  - Remove the single `QueryRemoteTool(config=config)` registration
  - For each provider in `config.remote_providers or []`: register
    `QueryRemoteTool(provider_cfg=provider, config=config)`
  - Tool names in the registry are now `query_<provider_name>` per source
- Add `remote_source_allowlist: Optional[set[str]] = None` class attribute to `BaseAgent`:
  - `None` means all available `query_<source>` tools are available to the agent
  - A non-None set lists the specific source names (not tool names) the agent may access
    (e.g., `{"brave_search", "semantic_scholar"}`)
- In `AgentHarness` tool filtering: when building the list of available tools for the LLM,
  filter per-source tools against the agent's `remote_source_allowlist`:
  - If `remote_source_allowlist is None`: all `query_<source>` tools pass through
  - If `remote_source_allowlist` is a set: only `query_<name>` where `name` is in the set pass through
- Update agents that previously had `query_remote` in their `tool_allowlist`:
  - Replace `"query_remote"` with appropriate `"query_<source_name>"` entries, or use
    `remote_source_allowlist` for more explicit control
- Run relevant tests: `pytest tests/agents/tools/test_registry.py tests/agents/test_harness.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/tools/__init__.py` (or registry file) — per-source instantiation
- `lsm/agents/base.py` — `remote_source_allowlist` class attribute
- `lsm/agents/harness.py` — tool filtering logic for per-source tools
- Any agent files that had `"query_remote"` in `tool_allowlist` — update
- `tests/agents/tools/test_registry.py` — update:
  - Test: registry contains `query_<name>` tool for each configured remote source
  - Test: registry does NOT contain a generic `query_remote` tool
  - Test: `remote_source_allowlist={"source_a"}` on agent causes only `query_source_a` to
    appear in the harness's available tool list; `query_source_b` is excluded
  - Test: `remote_source_allowlist=None` causes all `query_<source>` tools to be available

**Success criteria**: Tool registry contains one tool per remote source. `AgentHarness` filters
correctly based on `remote_source_allowlist`. No generic `query_remote` tool exists in the registry.

---

## 7.3: Update NewsAssistantAgent for New query_remote Design

**Description**: `NewsAssistantAgent` currently uses a `_resolve_lsm_config()` hack —
it looks up `query_remote` from the tool registry to obtain the config object. This hack
must be replaced since `query_remote` no longer exists as a single tool.

**Tasks**:
- Review `NewsAssistantAgent._resolve_lsm_config()`: it calls `self.tool_registry.lookup("query_remote")`
  and returns `tool.config` — this will fail after Phase 7.1
- Replace with direct config injection: add `lsm_config: LSMConfig` as a constructor parameter
  to `NewsAssistantAgent` (and similarly `CalendarAssistantAgent` and `EmailAssistantAgent`
  which have the same hack)
- Update all three agents' `_resolve_lsm_config()` to simply return `self.lsm_config`
- Update the agent factory to inject `lsm_config` at construction time
- Run relevant tests: `pytest tests/agents/ -v -k "news or calendar or email"`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/assistants/news_assistant.py` — inject `lsm_config`
- `lsm/agents/assistants/calendar_assistant.py` — inject `lsm_config`
- `lsm/agents/assistants/email_assistant.py` — inject `lsm_config`
- `lsm/agents/factory.py` — update construction to pass `lsm_config`
- `tests/agents/test_news_assistant_agent.py` — update construction
- `tests/agents/test_calendar_assistant_agent.py` — update construction
- `tests/agents/test_email_assistant_agent.py` — update construction

**Success criteria**: `NewsAssistantAgent`, `CalendarAssistantAgent`, and `EmailAssistantAgent`
no longer look up `query_remote` from the tool registry. Config is injected directly. All tests pass.

---

## 7.4: Phase 7 Code Review and Changelog

**Tasks**:
- Grep entire codebase for `"query_remote"` string — result must be empty outside of
  Phase 7 tests that assert its absence
- Grep entire codebase for `_find_provider` — result must be empty
- Verify all per-source `QueryRemoteTool` instances have unique names in the registry
- Verify `remote_source_allowlist=None` correctly exposes all sources
- Verify `remote_source_allowlist={"source_a"}` correctly excludes other sources
- Review `_resolve_lsm_config()` hack removal: confirm no agent looks up `query_remote` anymore
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 7 changes
- Update `.agents/docs/architecture/packages/lsm.agents.md`: document per-source
  `QueryRemoteTool` design and `remote_source_allowlist`
- Update `.agents/docs/architecture/api-reference/REMOTE.md` if it documents `query_remote`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/packages/lsm.agents.md`
- `.agents/docs/architecture/api-reference/REMOTE.md`

**Success criteria**: `pytest tests/ -v` passes. No references to `query_remote` as a single
generic tool. Per-source design is documented.
