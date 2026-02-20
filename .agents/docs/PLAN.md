# Local Second Mind v0.7.0 Development Plan: Agents and Remote Sources

## Overview

v0.7.0 introduces a comprehensive agent system with multiple specialized agents, advanced file tooling, execution environments, and remote source providers. This plan organizes work into 11 phased releases for incremental delivery with proper dependency ordering.

---

## Validation Notes (Architecture Alignment)

- Agents are already implemented under `lsm/agents/` with built-in agents (`curator`, `meta`, `research`, `writing`, `synthesis`) and a registry in `lsm/agents/factory.py`; new agents should extend this framework rather than replace it.
- Tooling and runners live in `lsm/agents/tools/` (`docker_runner.py`, `runner.py`, `ask_user.py`, `source_map.py`, etc.); new file tools and runners should be added here and wired through the ToolRegistry.
- Remote providers already include `openalex`, `crossref`, `core`, and `oai_pmh` under `lsm/remote/providers/`; v0.7.0 should extend these and add missing providers instead of duplicating them.
- Remote provider interface and chain behavior are defined in `.agents/docs/architecture/api-reference/REMOTE.md`; new providers must conform to `BaseRemoteProvider` and chain output schemas.

---

## Standard Phase Completion Checklist

*Applies to every phase — not repeated as individual tasks:*
- Review that no backwards compatibility and dead code remains
- Ensure test coverage for all new code (TDD: tests first, then implementation)
- Ensure the documentation in docs/ and .agents/docs/ are properly updated for changes.
- Update `docs/CHANGELOG.md` with phase deliverables

---

## Phase 1: Core Infrastructure & Configuration

**Why first:** Utils module and tiered model config are consumed by everything that follows. Logging changes and test harness shape all later agent work.

| Task | Description | Depends On |
|------|-------------|------------|
| 1.1 | Create `lsm.utils` module | None |
| 1.2 | Agent log format conversion | 1.1 |
| 1.3 | Tiered model configuration | None |
| 1.4 | Agent/tool test harness | None |

### 1.1: Create `lsm.utils` Module
- **Description:** Establish a shared utilities package for cross-cutting concerns currently scattered across modules.
- **Tasks:**
  - Create `lsm/utils/__init__.py` with public API.
  - Create `lsm/utils/logger.py` — centralized logger factory with configurable output (file, stream), format (plain text), and level (normal/verbose/debug).
  - Create `lsm/utils/paths.py` — common path resolution helpers (currently duplicated in config loader and agent harness).
  - Migrate existing utility code from `lsm/ingest/utils.py`, `lsm/agents/log_formatter.py`, and similar locations to use shared utilities where appropriate. Do NOT move ingest-specific code.
- **Files:**
  - `lsm/utils/__init__.py`
  - `lsm/utils/logger.py`
  - `lsm/utils/paths.py`
- **Success criteria:** Shared logger is importable from `lsm.utils`, existing tests pass, no duplication of path resolution logic.

### 1.2: Agent Log Format Conversion
- **Description:** Convert agent logs from JSON to plain text with configurable verbosity. Logs persist in agent workspace by default.
- **Tasks:**
  - Update `AgentLogEntry` serialization to emit plain text lines with timestamp, actor, and content.
  - Add verbosity levels: `normal` (actions and results only), `verbose` (+ tool arguments), `debug` (+ full LLM prompts/responses).
  - Ensure log files write to `<agents_folder>/<agent_name>/logs/` by default.
  - Update `log_formatter.py` and any JSON log consumers.
- **Files:**
  - `lsm/agents/log_formatter.py`
  - `lsm/agents/models.py`
  - `lsm/agents/harness.py`
- **Success criteria:** Agent runs produce plain text logs at configured verbosity, saved to workspace. Existing log-reading code (TUI log streaming) still works.

### 1.3: Tiered Model Configuration
- **Description:** Add quick/normal/complex model tiers so agents and tools can declare what LLM capability they need, with fallback resolution baked into config loading.
- **Tasks:**
  - Define tier schema in config: `llms.tiers: { quick: { provider, model }, normal: { ... }, complex: { ... } }`.
  - Each agent/tool declares its required tier (e.g., `tier = "quick"` for tagging, `tier = "complex"` for research synthesis).
  - During config loading, after `llms.services` are loaded, any service without an explicit model assignment inherits from its tier's default.
  - Update `resolve_service()` to incorporate tier fallback: explicit service config → tier default → "default" service.
  - Add validation: warn if a tier is referenced but not configured.
- **Files:**
  - `lsm/config/models/llm.py`
  - `lsm/config/loader.py`
  - `lsm/agents/base.py`
  - `lsm/agents/tools/base.py`
- **Success criteria:** Agents/tools can declare `tier = "quick"` and get the right model without explicit per-service config. Existing configs without tiers continue to work (tiers are optional).

### 1.4: Agent/Tool Test Harness
- **Description:** Design and implement a test harness for measuring agent and tool efficiency and detecting regressions across feature changes.
- **Tasks:**
  - Define benchmark task format: a reproducible scenario with inputs, expected outcomes, and scoring dimensions.
  - Define scoring metrics: token usage, tool call count, wall-clock time, output quality (manual or LLM-judged).
  - Define regression thresholds: acceptable variance per metric before flagging a regression.
  - Implement a minimal harness runner that can execute a benchmark task, record metrics, and compare against a baseline.
  - Add initial benchmark tasks for file operations (find, read, edit) as baseline for Phase 4 tooling comparisons.
- **Files:**
  - `tests/benchmarks/`
  - `tests/benchmarks/harness.py`
  - `tests/benchmarks/tasks/`
  - `.agents/docs/architecture/development/TESTING.md`
- **Success criteria:** Harness runner can execute benchmark tasks, persist results, and compare against baselines. Baseline metrics exist for current naive file operations.

---

## Phase 2: File Graphing System

**Why second:** File graphs are consumed by advanced tooling (Phase 4) and specific agents (Librarian, Manuscript Editor in Phase 6). Needs utils from Phase 1.

**Depends on:** Phase 1.1 (utils module)

| Task | Description | Depends On |
|------|-------------|------------|
| 2.1 | Graph schema and interfaces | 1.1 |
| 2.2 | Code file grapher | 2.1 |
| 2.3 | Text document grapher | 2.1 |
| 2.4 | Tool integration hooks | 2.2, 2.3 |
| 2.5 | Tests and fixtures | 2.2, 2.3 |

### 2.1: Graph Schema and Interfaces
- **Description:** Define the graph data model and public interfaces for structural output.
- **Tasks:**
  - Define node schema (type, name, span, depth, parent/child, metadata) and serialization format. Schema must support both code nodes (function, class, import, block) and text nodes (heading, paragraph, list). Unified `GraphNode` type with a `node_type` discriminator.
  - Specify deterministic ordering rules and stable IDs for node references.
  - Add caching strategy: hash file contents, cache graph by content hash. Invalidate on content change, not timestamp.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/source_map.py`
- **Success criteria:** Graph schema is documented and stable across runs for identical inputs.

### 2.2: Code File Grapher
- **Description:** Build a code-aware grapher that exposes functions/classes/blocks similar to tree-sitter outlines.
- **Tasks:**
  - Implement a parser strategy (tree-sitter when available, fallback heuristic parser otherwise).
  - Emit structural nodes with precise line/byte spans.
  - Normalize language-specific nodes into a common schema (function/class/block/import).
- **Files:**
  - `lsm/utils/file_graph.py`
- **Success criteria:** Code graphs expose accurate node boundaries across supported languages.

### 2.3: Text Document Grapher
- **Description:** Build a text grapher for markdown/plain text/docx headings and paragraphs.
- **Tasks:**
  - Parse headings and subheadings into a hierarchy.
  - Identify paragraph nodes with line spans under each heading.
  - Add docx parsing for section and paragraph extraction.
- **Package boundary note:** Common text processing logic (heading extraction, paragraph segmentation, section hierarchy building) should live in `lsm/utils/text_processing.py` — a new shared module. Both the ingest package (`structure_chunking.py`, parsers) and the file graph tools should import from this shared module. This avoids `lsm/utils/file_graph.py` depending on `lsm/ingest/` or vice versa. Shared data models (e.g., `TextSection`, `HeadingNode`) also belong in `lsm/utils/text_processing.py`.
- Existing `PageSegment` and `StructuredChunk` in `lsm/ingest/models.py` remain ingest-specific. The shared text processing module provides a parallel, graph-oriented representation that the ingest models can optionally wrap or convert from.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/utils/text_processing.py`
- **Success criteria:** Text graphs represent heading hierarchy and paragraph boundaries consistently.

### 2.4: Tool Integration Hooks
- **Description:** Expose graph outputs to tools for section-aware read/edit operations.
- **Tasks:**
  - Expose graph via a `get_file_graph(path) -> FileGraph` function in `lsm/utils/file_graph.py`. Tools call this function; they do not parse files themselves.
  - Ensure graph output can be requested per file and per section.
  - Line-hash generation: each `GraphNode` carries a `line_hash` computed from its content. This is consumed by the edit engine in Phase 4.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/source_map.py`
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/file_metadata.py`
- **Success criteria:** Tools can retrieve graph output without duplicating parsing logic.

### 2.5: Tests and Fixtures
- **Description:** Validate graph output determinism and section accuracy.
- **Tasks:**
  - Add fixtures for code and text files with expected graph outputs.
  - Add tests for stable ordering, span correctness, and cache hits.
- **Files:**
  - `tests/test_tools/`
  - `tests/fixtures/`
- **Success criteria:** Graph outputs match fixtures and remain stable across runs.

---

## Phase 3: Agent Framework Overhaul

**Why third:** Must restructure the agent package before adding new agents in Phase 6. Tool API standardization shapes how all agents interact with tools.

**Depends on:** Phase 1.3 (tiered model config for tier declarations)

| Task | Description | Depends On |
|------|-------------|------------|
| 3.1 | Agent package restructure | None |
| 3.2 | Workspace defaults and structure | 3.1 |
| 3.3 | Tool API standardization | None |
| 3.4 | Universal ask_user availability | None |

### 3.1: Agent Package Restructure
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

### 3.2: Workspace Defaults and Structure
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

### 3.3: Tool API Standardization
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

### 3.4: Universal ask_user Availability
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

---

## Phase 4: Advanced Tooling

**Why fourth:** Builds on file graphing (Phase 2) to create the find/read/edit tools that agents (Phase 6) will use. This is the "tilth-inspired" structural awareness tooling.

**Depends on:** Phase 1.4 (test harness for benchmarking), Phase 2 (file graphing), Phase 3.3 (tool API standardization)

**Harness requirement:** Every new tool in this phase must be benchmarked against the naive baseline (captured in Phase 1.4) using the test harness. Benchmarks must demonstrate that the graph-aware tools reduce tool call count and/or token usage compared to naive find/read/edit workflows. If a tool does not outperform the baseline, it must be redesigned or the naive approach kept.

| Task | Description | Depends On |
|------|-------------|------------|
| 4.1 | Tool API design and registry updates | 3.3 |
| 4.2 | Line-hash editing engine | 2.4 |
| 4.3 | Find file and find section tools | 2.4 |
| 4.4 | Read/outline enhancements | 2.4 |
| 4.5 | Benchmark comparisons | 1.4, 4.2–4.4 |
| 4.6 | Tests and documentation | 4.1–4.5 |

### 4.1: Tool API Design and Registry Updates
- **Description:** Define new tool surfaces and align them with ToolRegistry expectations.
- **Tasks:**
  - Review existing file tools (`read_file`, `write_file`, `source_map`) for extension points.
  - Define schemas for `find_file`, `find_section`, and `edit_file` tools.
  - Update tool metadata (risk level, runner preference, network needs).
- **Files:**
  - `lsm/agents/tools/base.py`
  - `lsm/agents/tools/**`
- **Success criteria:** Tool definitions are registered and discoverable with consistent schemas.

### 4.2: Line-Hash Editing Engine
- **Description:** Implement a deterministic edit engine that uses line hashes to precisely identify replacement targets. Motivated by reducing the multi-call "glob → read → too big → grep → read again" pattern that wastes agent tokens. See [blog.can.ac/2026/02/12/the-harness-problem](https://blog.can.ac/2026/02/12/the-harness-problem/) and [tilth](https://github.com/jahala/tilth) for prior art.
- **Tasks:**
  - Define line-hash format: short hash per line (e.g., first 8 chars of SHA-256 of line content).
  - `edit_file` tool accepts `{ file, start_hash, end_hash, new_content }` to replace a line range.
  - On hash mismatch: return descriptive error including the actual hashes at the target lines, surrounding context, and suggestions for the correct range. This enables intelligent retry by the LLM.
  - On success: return the updated file graph (outline) so the agent has current structural awareness.
  - Add collision detection: if the same hash appears multiple times, require additional disambiguation (line number or surrounding context).
- **Files:**
  - `lsm/agents/tools/edit_file.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Edits apply only when hashes match. Failures include actionable diagnostics. Edit + re-read is a single tool call.

### 4.3: Find File and Find Section Tools
- **Description:** Provide fast file and section discovery using structural graphs, so agents get structural awareness in one call instead of 6.
- **Tasks:**
  - `find_file` tool: search by name pattern, content pattern, or both. Returns file paths with brief structural outlines.
  - `find_section` tool: given a file (or searched files), find sections by heading/function/class name. Returns the section's graph node with line range and line hashes.
  - Graph-aware filters: language, node type (function/class/heading), depth limit.
  - Output includes line hashes for immediate use with `edit_file`.
- **Files:**
  - `lsm/agents/tools/find_file.py`
  - `lsm/agents/tools/find_section.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** An agent can find a function and get its content + line hashes in a single tool call.

### 4.4: Read/Outline Enhancements
- **Description:** Extend read tools to provide structural outlines and section-targeted reads.
- **Tasks:**
  - `read_file` accepts optional `section` parameter (heading name, function name, or graph node ID) to return only that section.
  - `read_file` accepts optional `max_depth` to control outline depth.
  - `source_map` returns structural outlines derived from file graphs instead of flat content.
  - All read outputs include line hashes when `include_hashes=true`.
- **Files:**
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/source_map.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Read tools can return section-only content and structural outlines.

### 4.5: Benchmark Comparisons
- **Description:** Use the test harness from 1.4 to validate that advanced tools outperform naive implementations.
- **Tasks:**
  - Define benchmark scenarios: find a function in a large codebase, edit a specific section, read an outline of a complex file.
  - Run each scenario with naive tools (current `read_file` + `write_file`) and record baseline metrics.
  - Run each scenario with advanced tools (line-hash edit, find_section, graph-aware read) and record metrics.
  - Compare: tool call count, total tokens consumed, wall-clock time, success rate.
  - Document results and flag any tools that do not demonstrate improvement.
- **Files:**
  - `tests/benchmarks/tasks/`
  - `tests/benchmarks/results/`
- **Success criteria:** Advanced tools demonstrate measurable improvement over naive baselines in at least tool call count and token usage.

### 4.6: Tests and Documentation
- **Description:** Validate tool behavior and document new usage.
- **Tasks:**
  - Add tests for find/read/edit flows using fixtures.
  - Document tool schemas and examples in `docs/`.
- **Files:**
  - `tests/test_tools/`
  - `docs/`
- **Success criteria:** Tooling works end-to-end with documented schemas and passing tests.

---

## Phase 5: Execution Environment

**Why fifth:** Runners and shell tools are independent of agent catalog but must be ready before agents that need code execution (Coder agent in Phase 6).

**Depends on:** Phase 3 (sandbox policy updates)

| Task | Description | Depends On |
|------|-------------|------------|
| 5.1 | Runner policy and config updates | None |
| 5.2 | Docker runner completion | 5.1 |
| 5.3 | WSL2 runner implementation | 5.1 |
| 5.4 | Bash and PowerShell tools | 5.1 |
| 5.5 | Tests and documentation | 5.2–5.4 |

### 5.1: Runner Policy and Config Updates
- **Description:** Align runner selection with sandbox policy and add configuration hooks for new runners.
- **Tasks:**
  - Audit current `ToolSandbox` runner selection rules.
  - Add configuration fields for WSL2 runner and command allow/deny lists.
  - Update runner selection to respect `execution_mode` and `force_docker` policy.
- **Files:**
  - `lsm/agents/tools/sandbox.py`
  - `lsm/config/**`
- **Success criteria:** Runner selection is policy-driven and configurable.

### 5.2: Docker Runner Completion
- **Description:** Finish Docker runner behavior and evaluate hot-load execution paths.
- **Tasks:**
  - Implement missing Docker runner features (volume mapping, environment scrubbing, timeout limits).
  - Evaluate hot-load strategy (persistent container with volume mounts) vs per-run container spawning. Document trade-offs and implement the chosen approach.
  - Volume mapping must respect sandbox `allowed_read_paths` and `allowed_write_paths`.
  - Environment scrubbing via existing `env_scrubber.py`.
  - Document runner constraints and failure modes.
- **Files:**
  - `lsm/agents/tools/docker_runner.py`
  - `lsm/agents/tools/env_scrubber.py`
  - `docs/`
- **Success criteria:** Docker runner executes tool commands with enforced limits and predictable lifecycle behavior.

### 5.3: WSL2 Runner Implementation
- **Description:** Add a Windows-hosted WSL2 runner for exec/network tools.
- **Tasks:**
  - Implement `WSL2Runner` with path translation and environment scrubbing.
  - Integrate runner availability checks in `ToolSandbox`.
  - Add logging and error handling for WSL2 invocation failures.
- **Files:**
  - `lsm/agents/tools/wsl2_runner.py`
  - `lsm/agents/tools/sandbox.py`
- **Success criteria:** WSL2 runner executes commands in sandbox constraints when enabled.

### 5.4: Bash and PowerShell Tools
- **Description:** Provide shell command tools with allow/deny constraints and path validation.
- **Tasks:**
  - Add `bash` tool for command execution with allow/deny config.
  - Add `powershell` tool for command execution with allow/deny config.
  - Implement path validation and sandbox enforcement for command arguments. Use `ToolSandbox.check_read_path()` / `check_write_path()` for any file path arguments detected in the command string.
  - Allow/deny configuration: `agents.sandbox.command_allowlist` and `agents.sandbox.command_denylist` in config.
- **Security testing strategy:** Shell tools are a high-risk attack surface. Security tests must cover:
  - Command injection via argument escaping (semicolons, pipes, backticks, `$()`, `&&`, `||`)
  - Path traversal attacks (relative paths, symlinks, `..` sequences escaping sandbox)
  - Environment variable exfiltration attempts
  - Allowlist/denylist bypass attempts (case variations, path aliasing, command aliasing)
  - Resource exhaustion (fork bombs, infinite loops, disk-filling commands)
  - Chained command sequences that individually pass but collectively escape sandbox
  - Tests must be added to the existing STRIDE security test suite (T1–T8 categories), specifically T1 (file access), T2 (command escalation), and T5 (resource exhaustion).
- **Files:**
  - `lsm/agents/tools/bash.py`
  - `lsm/agents/tools/powershell.py`
  - `lsm/agents/tools/sandbox.py`
  - `tests/test_agents/test_security_*.py`
- **Success criteria:** Shell tools honor allow/deny lists and validate file paths before execution. Security tests pass across all STRIDE categories.

### 5.5: Tests and Documentation
- **Description:** Validate runner and shell behavior with policy tests.
- **Tasks:**
  - Add unit tests for runner selection and command allow/deny behavior.
  - Add integration tests for Docker and WSL2 runners (when available).
  - Document config options and usage examples.
- **Files:**
  - `tests/test_agents_tools/`
  - `docs/`
- **Success criteria:** Runner and shell tools are tested and documented with clear configuration examples.

---

## Phase 6: Core Agent Catalog

**Why sixth:** Now that framework is restructured (Phase 3), tooling is enhanced (Phase 4), and runners are ready (Phase 5), new agents can be built on solid foundations.

**Depends on:** Phase 3.1 (restructure), Phase 4 (tooling), Phase 5 (runners for Coder)

| Task | Description | Depends On |
|------|-------------|------------|
| 6.1 | Agent catalog and registration | 3.1 |
| 6.2 | General Agent | 3.3, 4.1 |
| 6.3 | Librarian Agent | 2.4, 4.3 |
| 6.4 | Assistant Agent | None beyond base framework |
| 6.5 | Coder Agent | 4.2, 4.3, 5.2 or 5.3 |
| 6.6 | Manuscript Editor Agent | 2.3, 4.2, 4.4 |

### 6.1: Agent Catalog and Registration
- **Description:** Extend the existing agent registry to include new agent types and themes.
- **Tasks:**
  - Add new agent names to `lsm/agents/factory.py` and module exports.
  - Update UI/shell agent lists to include new types.
  - Ensure agent metadata (description, default tools, risk posture) is exposed.
- **Files:**
  - `lsm/agents/factory.py`
  - `lsm/ui/shell/commands/agents.py`
  - `lsm/ui/tui/screens/agents.py`
- **Success criteria:** New agents are discoverable in UI and shell commands.

### 6.2: General Agent
- **Description:** Provide a general-purpose task agent that uses available tools responsibly.
- **Tasks:**
  - Define prompt and tool allowlist defaults.
  - Implement run outputs (summary + artifact list).
  - Add guardrails for permission and iteration limits.
- **Files:**
  - `lsm/agents/productivity/general.py`
  - `lsm/agents/harness.py`
- **Success criteria:** General agent can execute tool-based tasks and emits standard artifacts.

### 6.3: Librarian Agent
- **Description:** Explore embeddings DB and create idea graphs/metadata summaries.
- **Tasks:**
  - Define embeddings query workflow and output format.
  - Integrate graph output with `file_graph` and memory tools.
  - Emit `idea_graph.md` and supporting artifacts.
- **Files:**
  - `lsm/agents/productivity/librarian.py`
  - `lsm/agents/tools/query_embeddings.py`
- **Success criteria:** Librarian agent produces a structured idea graph and citations.

### 6.4: Assistant Agent
- **Description:** Summarize cross-agent activity and suggest memory updates. This is the "HR for agents" assistant — NOT the communication assistants (those are in Phase 9).
- **Tasks:**
  - Review `run_summary.json` artifacts from all agent runs.
  - Produce consolidated summaries of actions taken, results produced, and security concerns.
  - Identify cross-agent patterns that should be stored as memories or trigger memory updates/removals.
  - Emit action recommendations for user review.
  - Integrate with memory tools for promotion/rejection workflow.
- **Files:**
  - `lsm/agents/assistants/assistant.py`
  - `lsm/agents/memory/**`
- **Success criteria:** Assistant agent produces consolidated summaries and actionable memory candidates.

### 6.5: Coder Agent
- **Description:** Specialized agent for code editing tasks.
- **Tasks:**
  - Define code-editing workflow with find/read/edit tooling: understand task → find relevant files → read sections → plan changes → edit with line-hash → verify.
  - Emit code change summaries, list of touched files, and diff artifact.
  - Enforce read/write scope based on sandbox rules.
  - Execute within sandbox via runner (Docker or WSL2).
- **Files:**
  - `lsm/agents/productivity/coder.py`
  - `lsm/agents/tools/**`
- **Success criteria:** Coder agent can plan, edit, and summarize code changes reliably.

### 6.6: Manuscript Editor Agent
- **Description:** Specialized agent for iteratively editing text documents.
- **Tasks:**
  - Define manuscript editing workflow: read document outline → identify sections for revision → iterative editing rounds → emit revision log + final artifact.
  - Use text graphing for section-level edits.
  - Emit revision logs and a final manuscript artifact.
- **Files:**
  - `lsm/agents/productivity/manuscript_editor.py`
  - `lsm/utils/file_graph.py`
- **Success criteria:** Manuscript editor produces revised documents with traceable edits.

---

## Phase 7: AI Providers & Protocol Infrastructure

**Why seventh:** OpenRouter and protocol work (OAI consolidation, MCP, RSS) must be in place before adding remote source providers in Phase 8.

**Depends on:** Phase 1.3 (tiered config), Phase 3.3 (function calling standardization)

| Task | Description | Depends On |
|------|-------------|------------|
| 7.1 | OpenRouter provider | 1.3 |
| 7.2 | OAI-protocol base class consolidation | None |
| 7.3 | MCP host support | 3.3 |
| 7.4 | Generic RSS reader with caching | None |

### 7.1: OpenRouter Provider
- **Description:** Add OpenRouter as an LLM provider, routing to any model available via OpenRouter's API.
- **Tasks:**
  - Implement `OpenRouterProvider` extending base LLM provider.
  - Support model selection via `provider_name: "openrouter"` in config with `model` field for specific model routing.
  - Handle OpenRouter-specific features: model fallbacks, prompt caching headers, usage tracking.
  - Add to `.env.example`: `OPENROUTER_API_KEY`.
- **Files:**
  - `lsm/providers/openrouter.py`
  - `lsm/config/models/llm.py`
  - `.env.example`
- **Success criteria:** OpenRouter provider works end-to-end for query and agent workflows.

### 7.2: OAI-Protocol Base Class Consolidation
- **Description:** Ensure all remote providers using OAI-PMH protocol derive from a shared `BaseOAIProvider` to reduce code duplication.
- **Tasks:**
  - Audit existing OAI-PMH providers: `arxiv.py`, `oai_pmh.py` — identify shared logic (record parsing, resumption tokens, metadata format mapping).
  - Extract shared OAI logic into `BaseOAIProvider` base class.
  - Refactor existing providers to inherit from `BaseOAIProvider`.
  - New OAI-based providers in Phase 8 must use this base.
- **Files:**
  - `lsm/remote/providers/base_oai.py`
  - `lsm/remote/providers/arxiv.py`
  - `lsm/remote/providers/oai_pmh.py`
- **Success criteria:** OAI providers share common harvesting logic. Adding a new OAI repository requires only config, not new parsing code.

### 7.3: MCP Host Support
- **Description:** Enable Local Second Mind to act as an MCP (Model Context Protocol) host, exposing MCP tools to agents (and later to query workflows).
- **Tasks:**
  - Implement MCP host client that can connect to MCP servers.
  - Register MCP-provided tools in the `ToolRegistry` so agents can discover and use them.
  - Add configuration as a **global config** field: `mcp_servers: [{ name, command, args, env }]` under the `"global"` section of config. This is global (not agent-specific) because MCP servers will also be usable from query workflows in a future version.
  - Ensure MCP tool execution respects sandbox constraints when used by agents.
  - Wire MCP server lifecycle management: start servers on demand, restart on failure, shutdown on application exit.
- **Files:**
  - `lsm/agents/tools/mcp_host.py`
  - `lsm/config/models/global_config.py`
  - `lsm/agents/tools/__init__.py`
- **Success criteria:** Agents can discover and invoke tools provided by external MCP servers. MCP tools appear in `ToolRegistry.list_tools()`. MCP config is in global section.

### 7.4: Generic RSS Reader
- **Description:** Implement a reusable RSS/Atom feed reader with caching for use by news providers and general feed consumption.
- **Tasks:**
  - Parse RSS 2.0 and Atom feeds.
  - Cache past reads with configurable TTL. Track seen items to download only new entries.
  - Normalize feed items to `RemoteResult` schema.
  - Expose as both a standalone remote provider (`rss` type) and a utility for other providers.
- **Files:**
  - `lsm/remote/providers/rss.py`
  - `lsm/remote/storage.py`
- **Success criteria:** RSS provider fetches feeds, caches results, and returns normalized items. Subsequent fetches return only new items.

---

## Phase 8: Remote Source Providers

**Why eighth:** Protocol infrastructure (OAI consolidation, RSS) from Phase 7 is now available. Providers are grouped by domain within the phase.

**Depends on:** Phase 7.2 (OAI consolidation), Phase 7.4 (RSS reader)

**Package structure:** Remote providers must be organized into domain-specific sub-packages under `lsm/remote/providers/`:
- `lsm/remote/providers/academic/` — scholarly, biomedical, philosophy, humanities
- `lsm/remote/providers/cultural/` — archives, museums, cultural heritage
- `lsm/remote/providers/news/` — news sources and RSS
- `lsm/remote/providers/finance/` — market data, government economic data
- `lsm/remote/providers/web/` — web search (Brave, etc.)
- Existing providers (`brave.py`, `wikipedia.py`, `arxiv.py`, etc.) must be migrated into their appropriate sub-packages.
- `lsm/remote/providers/__init__.py` re-exports for backwards compatibility.

| Task | Description | Depends On |
|------|-------------|------------|
| 8.0 | Provider sub-package restructure | None |
| 8.1 | Structured output validation framework | None |
| 8.2 | Scholarly discovery pipeline | 7.2, 8.1 |
| 8.3 | Academic & biomedical providers | 7.2, 8.1 |
| 8.4 | Cultural heritage & archive providers | 8.1 |
| 8.5 | News providers | 7.4, 8.1 |
| 8.6 | Financial & government data providers | 8.1 |
| 8.7 | Specialized protocol providers | 8.1 |
| 8.8 | Tests and documentation | 8.0–8.7 |

### 8.0: Provider Sub-Package Restructure
- **Description:** Reorganize existing and new remote providers into domain-specific sub-packages.
- **Tasks:**
  - Create sub-package directories: `academic/`, `cultural/`, `news/`, `finance/`, `web/`.
  - Move existing providers into appropriate sub-packages (e.g., `brave.py` → `web/brave.py`, `arxiv.py` → `academic/arxiv.py`, `openalex.py` → `academic/openalex.py`, `wikipedia.py` → `web/wikipedia.py`).
  - Update `lsm/remote/providers/__init__.py` re-exports and factory registration so existing imports and config references continue to work.
  - Update all test imports.
- **Files:**
  - `lsm/remote/providers/*/`
  - `lsm/remote/__init__.py`
  - `lsm/remote/factory.py`
  - `tests/test_providers/remote/`
- **Success criteria:** All existing provider tests pass from new locations. Factory registration unchanged. Config references unchanged.

### 8.1: Structured Output Validation Framework
- **Description:** Ensure all remote providers produce stable, structured output suitable for use in `RemoteProviderChain`. This is a prerequisite for all provider implementations.
- **Tasks:**
  - Define a structured output contract: every provider's `search()` and `search_structured()` must return `RemoteResult` objects with all required fields populated (title, url, snippet, score, metadata with stable IDs).
  - Add a `validate_output(results: List[RemoteResult]) -> List[str]` utility that checks output conformance and returns a list of violations.
  - Add `get_output_fields()` contract enforcement: every provider must declare its output fields, and `RemoteProviderChain` must validate that a link's output fields match the next link's expected input fields.
  - Add integration test base class `RemoteProviderOutputTest` that all provider tests can inherit from to automatically validate output structure.
  - Audit and update all existing providers to conform to the validated output contract.
- **Files:**
  - `lsm/remote/base.py`
  - `lsm/remote/chain.py`
  - `lsm/remote/validation.py`
  - `tests/test_providers/remote/test_base.py`
- **Success criteria:** All providers pass output validation. Chain field mapping is validated at construction time. New providers automatically inherit output validation tests.

### 8.2: Scholarly Discovery Pipeline
- **Description:** Chain OpenAlex → Crossref → Unpaywall → CORE to discover, enrich, resolve open-access links, and retrieve full-text scholarly documents.
- **Tasks:**
  - Extend existing `openalex.py` to return stable IDs and DOI metadata in structured output.
  - Extend existing `crossref.py` to enrich metadata and resolve DOI/ISSN.
  - Implement `unpaywall.py` provider for open-access link resolution by DOI.
  - Wire a `RemoteProviderChain` that downloads full text when available and hands off to ingest pipeline.
  - Update existing `core.py` as the full-text retrieval fallback.
- **Files:**
  - `lsm/remote/providers/academic/openalex.py`
  - `lsm/remote/providers/academic/crossref.py`
  - `lsm/remote/providers/academic/unpaywall.py`
  - `lsm/remote/providers/academic/core.py`
  - `lsm/remote/chain.py`
- **Success criteria:** A DOI query yields normalized metadata + full-text retrieval when available. Chain executes end-to-end. All providers pass structured output validation (8.1).

### 8.3: Academic & Biomedical Providers
- **Description:** Add discipline-specific academic sources.
- **Providers:**
  - `pubmed.py` — PubMed/PubMed Central via E-utilities. Open-access full text when available.
  - `ssrn.py` — SSRN preprints and metadata.
  - `philarchive.py` — PhilArchive philosophy preprints.
  - `project_muse.py` — Project MUSE humanities metadata.
- **Files:**
  - `lsm/remote/providers/academic/pubmed.py`
  - `lsm/remote/providers/academic/ssrn.py`
  - `lsm/remote/providers/academic/philarchive.py`
  - `lsm/remote/providers/academic/project_muse.py`
- **Success criteria:** Each provider returns normalized `RemoteResult` with open-access links when possible. Passes output validation.

### 8.4: Cultural Heritage & Archive Providers
- **Description:** Add providers for archives, museums, and cultural heritage datasets.
- **Providers:**
  - `archive_org.py` — Archive.org metadata + file retrieval.
  - `dpla.py` — Digital Public Library of America.
  - `loc.py` — Library of Congress JSON/YAML API.
  - `smithsonian.py` — Smithsonian Open Access API.
  - `met.py` — Metropolitan Museum of Art Collection API.
  - `rijksmuseum.py` — Rijksmuseum data services.
  - `iiif.py` — IIIF Image/Presentation/Content Search APIs.
  - `wikidata.py` — Wikidata SPARQL endpoint.
- **Files:**
  - `lsm/remote/providers/cultural/archive_org.py`
  - `lsm/remote/providers/cultural/dpla.py`
  - `lsm/remote/providers/cultural/loc.py`
  - `lsm/remote/providers/cultural/smithsonian.py`
  - `lsm/remote/providers/cultural/met.py`
  - `lsm/remote/providers/cultural/rijksmuseum.py`
  - `lsm/remote/providers/cultural/iiif.py`
  - `lsm/remote/providers/cultural/wikidata.py`
- **Success criteria:** Each provider returns normalized results with stable IDs and source URLs. Passes output validation.

### 8.5: News Providers
- **Description:** Implement news sources with API-backed retrieval. Sources without APIs use the RSS reader from 7.4.
- **Providers:**
  - `nytimes.py` — NYTimes Top Stories + Article Search APIs.
  - `guardian.py` — The Guardian Content API (or GDELT for global coverage).
  - `newsapi.py` — NewsAPI for topic aggregation.
  - RSS feeds for sources without dedicated APIs (via `rss.py` provider).
- **Files:**
  - `lsm/remote/providers/news/nytimes.py`
  - `lsm/remote/providers/news/guardian.py`
  - `lsm/remote/providers/news/gdelt.py`
  - `lsm/remote/providers/news/newsapi.py`
- **Success criteria:** News queries return current articles with source, timestamp, and canonical URL. Passes output validation.

### 8.6: Financial & Government Data Providers
- **Description:** Provide market data, economic indicators, and government datasets from US, European, and international sources.
- **Providers:**
  - `alpha_vantage.py` or `iex_cloud.py` — US/global market data.
  - `fred.py` — Federal Reserve Economic Data (FRED) time series.
  - `sec_edgar.py` — SEC filings metadata and document retrieval.
  - `data_gov.py` — US federal datasets (Data.gov or Treasury).
  - `ecb.py` — European Central Bank Statistical Data Warehouse (SDW) API for Eurozone economic data.
  - `eurostat.py` — Eurostat for EU-wide statistics and indicators.
  - `world_bank.py` — World Bank Open Data API for global development indicators.
  - `imf.py` — IMF Data API for international financial statistics.
- **Files:**
  - `lsm/remote/providers/finance/alpha_vantage.py`
  - `lsm/remote/providers/finance/fred.py`
  - `lsm/remote/providers/finance/sec_edgar.py`
  - `lsm/remote/providers/finance/data_gov.py`
  - `lsm/remote/providers/finance/ecb.py`
  - `lsm/remote/providers/finance/eurostat.py`
  - `lsm/remote/providers/finance/world_bank.py`
  - `lsm/remote/providers/finance/imf.py`
- **Success criteria:** Financial queries return structured data with timestamps, source attribution, and currency/unit metadata where applicable. Passes output validation.

### 8.7: Specialized Protocol Providers
- **Providers:**
  - `perseus_cts.py` — Perseus CTS API for classical text retrieval by CTS URNs.
- **Files:**
  - `lsm/remote/providers/cultural/perseus_cts.py`
- **Success criteria:** CTS URN queries return text passages with citation metadata. Passes output validation.

### 8.8: Tests and Documentation
- **Description:** Validate provider behavior, authentication, and schema normalization.
- **Tasks:**
  - Add tests for provider config validation and output normalization.
  - Add integration tests for at least one provider per category.
  - Document API keys and configuration in `docs/` and `.env.example`.
- **Files:**
  - `tests/test_remote/`
  - `docs/`
  - `.env.example`
  - `example_config.json`
- **Success criteria:** Provider integrations are tested and documented with real config examples.

---

## Phase 9: Communication Platform

**Why ninth:** Email/calendar/news assistants need their data providers built alongside them. OAuth infrastructure is a prerequisite for email/calendar providers. Placed after remote sources (Phase 8) to leverage established provider patterns.

**Depends on:** Phase 6.4 (Assistant agent pattern), Phase 8 (provider patterns established)

| Task | Description | Depends On |
|------|-------------|------------|
| 9.1 | OAuth2 shared infrastructure | None |
| 9.2 | Email providers | 9.1 |
| 9.3 | Calendar providers | 9.1 |
| 9.4 | Email Assistant agent | 9.2, 6.1 |
| 9.5 | Calendar Assistant agent | 9.3, 6.1 |
| 9.6 | News Assistant agent | 8.5, 6.1 |

### 9.1: OAuth2 Shared Infrastructure
- **Description:** Build a shared OAuth2 client module for providers that require user authorization (Gmail, Google Calendar, Microsoft Graph).
- **Tasks:**
  - Implement OAuth2 authorization code flow with redirect handling (local HTTP callback server).
  - Token storage: encrypted tokens in `<global_folder>/oauth_tokens/` with per-provider isolation.
  - Automatic token refresh with configurable refresh buffer.
  - Consent/scope management: request minimal scopes, store granted scopes.
  - Add configuration: `remote_providers.<name>.oauth: { client_id, client_secret, scopes, redirect_uri }`.
- **Files:**
  - `lsm/remote/oauth.py`
  - `lsm/config/models/modes.py`
- **Success criteria:** OAuth flow completes for Google and Microsoft. Tokens persist and auto-refresh. Credentials are never logged.

### 9.2: Email Providers
- **Description:** Add major email providers for assistant agents.
- **Providers:**
  - `gmail.py` — Gmail via Google API (OAuth2). Read, search, draft.
  - `microsoft_graph_mail.py` — Outlook via Microsoft Graph (OAuth2). Read, search, draft.
  - `imap.py` — IMAP/SMTP fallback for self-hosted email.
- **Tasks:** All write operations (send, move, delete) must go through approval gating via `ask_user`.
- **Files:**
  - `lsm/remote/providers/gmail.py`
  - `lsm/remote/providers/microsoft_graph_mail.py`
  - `lsm/remote/providers/imap.py`
- **Success criteria:** Email providers can read and search mail. Draft creation requires explicit user approval before sending.

### 9.3: Calendar Providers
- **Description:** Add major calendar providers for assistant agents.
- **Providers:**
  - `google_calendar.py` — Google Calendar API (OAuth2).
  - `microsoft_graph_calendar.py` — Microsoft Graph Calendar (OAuth2).
  - `caldav.py` — CalDAV fallback for self-hosted calendars.
- **Tasks:** All mutating operations (create, update, delete events) require explicit user approval.
- **Files:**
  - `lsm/remote/providers/google_calendar.py`
  - `lsm/remote/providers/microsoft_graph_calendar.py`
  - `lsm/remote/providers/caldav.py`
- **Success criteria:** Calendar providers can read events and propose changes that require approval.

### 9.4: Email Assistant Agent
- **Description:** Agent that reads, summarizes, and drafts emails with user approval gating.
- **Tasks:**
  - Read emails by time window: last 1 hour, last 24 hours, custom range.
  - Filter by criteria: search string, to/from specific persons, unread only, specific folder.
  - Produce email summary organized by importance/topic.
  - Generate task list from action-requiring emails.
  - Draft reply/compose with explicit user approval before any send.
- **Files:**
  - `lsm/agents/assistants/email_assistant.py`
- **Success criteria:** Agent summarizes inbox, generates task lists, and drafts emails that require user approval to send.

### 9.5: Calendar Assistant Agent
- **Description:** Agent that reads calendar, provides scheduling intelligence, and manages events with approval.
- **Tasks:**
  - Read calendar and summarize upcoming events by day/week.
  - Given a proposed event, suggest available time slots based on existing calendar.
  - Add/remove/edit events with explicit user approval for every mutation.
- **Files:**
  - `lsm/agents/assistants/calendar_assistant.py`
- **Success criteria:** Agent provides scheduling suggestions and requires approval for all calendar changes.

### 9.6: News Assistant Agent
- **Description:** Agent that produces curated news summaries from configured news sources.
- **Tasks:**
  - Produce news summary in newsletter style over a configurable time frame.
  - Filter by specific topics or criteria.
  - Source from news providers (8.5) and RSS feeds.
- **Files:**
  - `lsm/agents/assistants/news_assistant.py`
- **Success criteria:** Agent produces newsletter-style summaries from configured news sources with topic filtering.

---

## Phase 10: Meta-Agent Evolution

**Why tenth:** Parallel execution builds on the existing serial meta-agent (already in `lsm/agents/meta.py`) and requires the full agent catalog (Phase 6) and communication assistants (Phase 9) to be available for orchestration.

**Depends on:** Phase 6 (core agents), Phase 9 (communication assistants)

| Task | Description | Depends On |
|------|-------------|------------|
| 10.1 | Parallel task graph planning | None |
| 10.2 | Parallel execution engine | 10.1 |
| 10.3 | General Meta-Agent | 10.2 |
| 10.4 | Assistant Meta-Agent | 10.2, 6.4 |
| 10.5 | Tests and documentation | 10.1–10.4 |

### 10.1: Parallel Task Graph Planning
- **Description:** Extend task graph planning to represent parallelizable work.
- **Tasks:**
  - Extend existing `task_graph.py` to support `parallel_group` nodes. A parallel group contains tasks that can execute concurrently. Dependency gates between groups enforce ordering.
  - Add deterministic ordering for parallel execution plans (within parallel groups for reproducible results).
  - Document graph serialization for meta-agent prompts.
- **Files:**
  - `lsm/agents/meta/task_graph.py`
  - `lsm/agents/meta/meta.py`
- **Success criteria:** Meta plans include explicit parallel groups with deterministic ordering.

### 10.2: Parallel Execution Engine
- **Description:** Execute sub-agents concurrently with resource limits and sandbox guarantees.
- **Tasks:**
  - Add concurrency controls using `asyncio` or `concurrent.futures` for parallel sub-agent execution.
  - Enforce sandbox monotonicity: sub-agent sandboxes must be subsets of the parent meta-agent sandbox.
  - Resource limits: respect `max_concurrent` from `AgentConfig`.
  - Collect artifacts concurrently and merge results deterministically (sorted by agent name, then task order).
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/tools/spawn_agent.py`
  - `lsm/agents/tools/await_agent.py`
  - `lsm/agents/tools/collect_artifacts.py`
- **Success criteria:** Parallel runs complete safely with predictable artifact aggregation.

### 10.3: General Meta-Agent
- **Description:** Provide a general meta-agent that orchestrates multiple agent types.
- **Tasks:**
  - Define meta-agent prompt and tool allowlist.
  - Implement sub-agent selection logic based on task types.
  - Emit final consolidated artifact (`final_result.md`).
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/factory.py`
- **Success criteria:** General meta-agent can coordinate multi-agent workflows end-to-end.

### 10.4: Assistant Meta-Agent
- **Description:** Provide a meta-agent dedicated to assistant-style summaries and compliance checks.
- **Tasks:**
  - Define assistant-meta prompt for reviews, security checks, and memory proposals.
  - Add optional validation passes over sub-agent outputs.
  - Emit summary artifacts and action recommendations.
- **Files:**
  - `lsm/agents/meta/meta.py`
  - `lsm/agents/assistants/assistant.py`
- **Success criteria:** Assistant meta-agent produces structured reviews and action items.

### 10.5: Tests and Documentation
- **Description:** Validate parallel meta-agent behavior and sandbox enforcement.
- **Tasks:**
  - Add tests for parallel plan execution and sandbox monotonicity.
  - Document meta-agent configuration and usage.
- **Files:**
  - `tests/test_agents_meta/`
  - `docs/`
- **Success criteria:** Meta-agent parallel execution is tested and documented.

---

## Phase 11: Release & Documentation

**Depends on:** All previous phases

| Task | Description |
|------|-------------|
| 11.1 | Version bump to 0.7.0 |
| 11.2 | Documentation audit |
| 11.3 | Config examples update |
| 11.4 | TUI WHATS_NEW |

### 11.1: Version Bump
- **Description:** Update version metadata to v0.7.0.
- **Tasks:**
  - Update `pyproject.toml`, `lsm/__init__.py`, and any runtime version references to `0.7.0`.
- **Files:**
  - `pyproject.toml`
  - `lsm/__init__.py`
- **Success criteria:** All version references are `0.7.0`.

### 11.2: Documentation Audit
- **Description:** Confirm documentation reflects the release scope.
- **Tasks:**
  - Verify all new agent, tool, and provider capabilities are documented.
  - Verify architecture docs reflect restructured agent packages and provider sub-packages.
  - Verify security docs cover new tools (bash, powershell, MCP) and new providers (OAuth-gated).
- **Files:**
  - `docs/**`
  - `.agents/docs/**`
- **Success criteria:** Documentation matches implemented changes.

### 11.3: Config Examples Update
- **Description:** Ensure examples include new config keys and API variables.
- **Tasks:**
  - Update `example_config.json` with: tier config, MCP server config, new provider entries, OAuth config, command allow/deny lists.
  - Update `.env.example` with new API keys (OpenRouter, news APIs, financial APIs, OAuth client IDs).
- **Files:**
  - `example_config.json`
  - `.env.example`
- **Success criteria:** Config examples are complete and match implemented configuration schema.

### 11.4: TUI WHATS_NEW
- **Description:** Update TUI with v0.7.0 highlights.
- **Tasks:**
  - Update `lsm/ui/tui/screens/help.py` WHATS_NEW section with v0.7.0 highlights.
- **Files:**
  - `lsm/ui/tui/screens/help.py`
  - `tests/test_ui/tui/test_screens.py`
- **Success criteria:** WHATS_NEW displays v0.7.0 features.

---

## Cross-Phase Dependency Summary

```
Phase 1  ──→  Phase 2  ──→  Phase 4  ──→  Phase 6  ──→  Phase 10  ──→  Phase 11
  │    │         │                            │               ↑
  │    │         └────────────────────────────┘               │
  │    │                                                      │
  │    └──→  Phase 3  ──→  Phase 5  ──→  Phase 6              │
  │                                        │                  │
  └──(1.4)──→  Phase 4  (benchmarks)       └──→  Phase 9  ──→  Phase 10
                                                    ↑
                                          Phase 7  ──→  Phase 8
```

**Key gates:**
- Phase 1.4 (test harness) must finish before Phase 4 (benchmark comparisons)
- Phase 2 (graphing) must finish before Phase 4 (tooling that uses graphs)
- Phase 3 (restructure) must finish before Phase 6 (new agents placed in new packages)
- Phase 7 (OAI consolidation, RSS) must finish before Phase 8 (providers that use them)
- Phase 8.1 (structured output validation) must finish before all provider implementations
- Phase 9 (communication platform) needs both Phase 8 (provider patterns) and Phase 6 (agent patterns)
- Phase 10 (meta-agents) needs Phase 6 + Phase 9 (agents to orchestrate)

**Parallelizable work:**
- Phases 2 and 3 can run in parallel (no mutual dependencies)
- Phase 5 can run in parallel with Phase 4 (independent concerns)
- Phase 7 can start as soon as Phase 1.3 is done, in parallel with Phases 4-6
