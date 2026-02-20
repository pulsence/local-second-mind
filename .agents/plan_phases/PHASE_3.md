# Phase 3: Agent Framework Overhaul

**Why third:** Must restructure the agent package before adding new agents in Phase 6. Tool API standardization shapes how all agents interact with tools.

**Depends on:** Phase 1.3 (tiered model config for tier declarations)

| Task | Description | Depends On |
|------|-------------|------------|
| 3.1 | Agent package restructure | None |
| 3.2 | Workspace defaults and structure | 3.1 |
| 3.3 | Tool API standardization | None |
| 3.4 | Universal ask_user availability | None |

## 3.1: Agent Package Restructure
- **Description:** Reorganize agents into thematic sub-packages for discoverability and scalability.
- **Tasks:**
  - Create sub-packages with the following mapping:
    - `lsm/agents/academic/` — `research.py`, `synthesis.py`, `curator.py` (existing agents, moved)
    - `lsm/agents/assistants/` — (empty initially; populated in Phases 6 and 9)
    - `lsm/agents/productivity/` — `writing.py` (existing, moved), plus new agents in Phase 6
    - `lsm/agents/meta/` — `meta.py` (existing, moved), `task_graph.py` (existing, moved)
  - Update `factory.py` to discover agents from sub-packages. Keep single `AgentRegistry`.
  - Update `__init__.py` re-exports so existing imports (`from lsm.agents import ResearchAgent`) continue to work.
  - Add `theme` and `category` metadata to agent registry entries for UI grouping.
  - Update UI agent lists (`lsm/ui/shell/commands/agents.py`, `lsm/ui/tui/screens/agents.py`) to display agents grouped by theme.
- **Files:**
  - `lsm/agents/academic/`
  - `lsm/agents/productivity/`
  - `lsm/agents/meta/`
  - `lsm/agents/factory.py`
  - `lsm/agents/__init__.py`
  - `lsm/ui/shell/commands/agents.py`
  - `lsm/ui/tui/screens/agents.py`
- **Success criteria:** Existing agents work from new locations. `from lsm.agents import ResearchAgent` still works. UI shows agents grouped by theme. All existing agent tests pass.

## 3.2: Workspace Defaults and Structure
- **Description:** Define and enforce a standard workspace directory layout for all agents.
- **Tasks:**
  - Default workspace per agent: `<agents_folder>/<agent_name>/` with sub-dirs: `logs/`, `artifacts/`, `memory/`.
  - Agent harness creates workspace structure on first run if it doesn't exist.
  - Agent `read_file`/`write_file` tools default to workspace paths when no absolute path given.
  - Document workspace structure in agent architecture docs.
- **Files:**
  - `lsm/agents/harness.py`
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/write_file.py`
  - `.agents/docs/architecture/development/AGENTS.md`
- **Success criteria:** New agent runs auto-create workspace dirs. File tools resolve relative paths within workspace.

## 3.3: Tool API Standardization
- **Description:** Use provider-native function calling API (`tools=[...]` parameter) when available; fall back to system-prompt tool descriptions when not.
- **Tasks:**
  - Update `AgentHarness._call_llm()` to pass `tools` parameter to providers that support function calling (OpenAI, Anthropic, Google).
  - For providers without native function calling support, serialize tool definitions into a human-readable block in the system prompt that mirrors the function-calling JSON schema.
  - Add `supports_function_calling` property to LLM provider base class.
  - Update tool response parsing to handle both native function call responses and text-based tool invocations.
- **Files:**
  - `lsm/agents/harness.py`
  - `lsm/providers/base.py`
  - `lsm/providers/*.py`
- **Success criteria:** Agents using OpenAI/Anthropic/Google providers use native function calling. Agents using other providers fall back to text-based tool descriptions. No change in agent behavior.

## 3.4: Universal ask_user Availability
- **Description:** Ensure `ask_user` tool is always available to every agent regardless of tool allowlist configuration.
- **Tasks:**
  - Already partially implemented (`_always_available_tools = {"ask_user"}` in `BaseAgent`). Verify this is enforced in all code paths.
  - Add `ignore_and_continue` configuration option: when enabled, `ask_user` calls are auto-responded with a "continue with your best judgment" message instead of prompting the user.
  - Add config field: `agents.interaction.auto_continue: bool = false`.
- **Files:**
  - `lsm/agents/base.py`
  - `lsm/agents/tools/ask_user.py`
  - `lsm/config/models/agents.py`
- **Success criteria:** `ask_user` works in all agents. `auto_continue` mode skips user prompts.
