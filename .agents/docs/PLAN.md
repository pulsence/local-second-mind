# Local Second Mind v0.7.0 Development Plan: Agents and Remote Sources

## Overview

v0.7.0 introduces a comprehensive agent system with multiple specialized agents and remote source providers. This plan organizes work into phased releases for incremental delivery.

---

## Validation Notes (Architecture Alignment)

- Agents are already implemented under `lsm/agents/` with built-in agents (`curator`, `meta`, `research`, `writing`, `synthesis`) and a registry in `lsm/agents/factory.py`; new agents should extend this framework rather than replace it.
- Tooling and runners live in `lsm/agents/tools/` (`docker_runner.py`, `runner.py`, `ask_user.py`, `source_map.py`, etc.); new file tools and runners should be added here and wired through the ToolRegistry.
- Remote providers already include `openalex`, `crossref`, `core`, and `oai_pmh` under `lsm/remote/providers/`; v0.7.0 should extend these and add missing providers instead of duplicating them.
- Remote provider interface and chain behavior are defined in `.agents/docs/architecture/api-reference/REMOTE.md`; new providers must conform to `BaseRemoteProvider` and chain output schemas.

---

## Phase 1: Core Architecture Foundations

### 1.1: Shared Utilities and Logging
- **Description:** Establish common utilities and logging conventions used across agents, tools, and providers.
- **Tasks:**
  - Create `lsm.utils` module for reusable helpers (logger, config helpers, common paths).
  - Convert agent logs to plain text with log levels (normal/verbose/debug) and ensure log storage under agent workspace.
  - Update any existing logging integration to use the centralized utility logger.
- **Files:**
  - `lsm/utils/__init__.py`
  - `lsm/utils/logger.py`
  - `lsm/agents/**`
  - `lsm/ui/**`
- **success criteria:** Agent logs are plain text with selectable verbosity, use shared logger, and persist under the agent workspace by default.

### 1.2: File Graphing Utility
- **Description:** Provide structural graph generation for code and text documents to support advanced find/edit tooling.
- **Tasks:**
  - Define graph schema and interfaces.
  - Implement code and text graphers.
  - Integrate graph outputs with tools and add tests.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/**`
  - `lsm/ingest/**`
  - `tests/**`
- **success criteria:** Graph output is deterministic, tool-consumable, and validated with fixtures.

#### 1.2.1: Graph Schema and Interfaces
- **Description:** Define the graph data model and public interfaces for structural output.
- **Tasks:**
  - Define node schema (type, name, span, depth, parent/child, metadata) and serialization format.
  - Specify deterministic ordering rules and stable IDs for node references.
  - Add caching strategy for repeated graph reads of unchanged files.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/agents/tools/source_map.py`
- **success criteria:** Graph schema is documented and stable across runs for identical inputs.

#### 1.2.2: Code File Grapher
- **Description:** Build a code-aware grapher that exposes functions/classes/blocks similar to tree-sitter outlines.
- **Tasks:**
  - Implement a parser strategy (tree-sitter when available, fallback heuristic parser otherwise).
  - Emit structural nodes with precise line/byte spans.
  - Normalize language-specific nodes into a common schema (function/class/block/import).
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/ingest/**`
- **success criteria:** Code graphs expose accurate node boundaries across supported languages.

#### 1.2.3: Text Document Grapher
- **Description:** Build a text grapher for markdown/plain text/docx headings and paragraphs.
- **Tasks:**
  - Parse headings and subheadings into a hierarchy.
  - Identify paragraph nodes with line spans under each heading.
  - Add docx parsing for section and paragraph extraction.
  - Respect existing ingest parsing behaviors for markdown, plaintext, and docx.
- **Files:**
  - `lsm/utils/file_graph.py`
  - `lsm/ingest/**`
- **success criteria:** Text graphs represent heading hierarchy and paragraph boundaries consistently.

#### 1.2.4: Tool Integration Hooks
- **Description:** Expose graph outputs to tools for section-aware read/edit operations.
- **Tasks:**
  - Add graph access APIs to `source_map` and new file tools.
  - Ensure graph output can be requested per file and per section.
  - Add integration points for line-hash generation.
- **Files:**
  - `lsm/agents/tools/source_map.py`
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/file_metadata.py`
- **success criteria:** Tools can retrieve graph output without duplicating parsing logic.

#### 1.2.5: Tests and Fixtures
- **Description:** Validate graph output determinism and section accuracy.
- **Tasks:**
  - Add fixtures for code and text files with expected graph outputs.
  - Add tests for stable ordering, span correctness, and cache hits.
- **Files:**
  - `tests/test_tools/`
  - `tests/fixtures/`
- **success criteria:** Graph outputs match fixtures and remain stable across runs.

### 1.3: Storage Research and Harness Plan
- **Description:** Validate storage paths and build a test harness plan for agent/tool regression tracking.
- **Tasks:**
  - Document SQLite vs PostgreSQL usage options for agent memory and tool data.
  - Define test harness spec for measuring agent/tool improvements and regressions.
- **Files:**
  - `.agents/docs/architecture/development/AGENTS.md`
  - `.agents/docs/architecture/development/TESTING.md`
  - `docs/`
- **success criteria:** Documentation includes a storage decision matrix and a concrete harness plan.

### 1.4: Phase 1 Review and Changelog
- **Description:** Finalize Phase 1 with code review and documentation updates.
- **Tasks:**
  - Review changes for backwards compatibility and dead code.
  - Ensure tests exist for new utilities and graph output.
  - Update `docs/CHANGELOG.md` for Phase 1 items.
- **Files:**
  - `docs/CHANGELOG.md`
  - `tests/**`
- **success criteria:** Phase 1 changes are tested, reviewed, and documented in the changelog.

## Phase 2: Agent Framework and Tooling

### 2.1: Agent Package Restructure and Workspace Defaults
- **Description:** Organize agent modules by theme and enforce workspace defaults.
- **Tasks:**
  - Restructure agents into thematic packages (e.g., `lsm.agents.academic`, `lsm.agents.assistants`, `lsm.agents.productivity`).
  - Ensure default agent workspace read/write access and log persistence.
  - Extend agent registry metadata for visible agent types (theme/category labels).
- **Files:**
  - `lsm/agents/**`
  - `lsm/agents/factory.py`
  - `lsm/config/**`
- **success criteria:** Agents are grouped by theme, default workspace rules apply, and agent types are discoverable.

### 2.2: Agent Tool Interface and Question Tool
- **Description:** Standardize tool exposure and allow clarification prompts for all agents.
- **Tasks:**
  - Update sending tools to use AI Model API field `tools=[...]` (function calling API) where possible.
  - When the `tools` field is not available, emit a human-readable list that mimics the same JSON structure used for function calling.
  - Ensure `ask_user` is available to all agents with a policy to ignore and continue when configured.
- **Files:**
  - `lsm/agents/tools/**`
  - `lsm/providers/**`
  - `lsm/agents/harness/**`
- **success criteria:** Agents can always request clarification and tool metadata is transmitted consistently.

### 2.3: Tooling Enhancements (Find/Read/Write/Edit)
- **Description:** Build advanced file manipulation tools using structural graphs and line-hash editing.
- **Tasks:**
  - Define tool APIs and registry updates for find/read/edit flows.
  - Implement line-hash editing and graph-aware discovery.
  - Document and test new tool behaviors.
- **Files:**
  - `lsm/agents/tools/**`
  - `lsm/utils/file_graph.py`
  - `docs/`
  - `tests/**`
- **success criteria:** Tools provide structural output in fewer calls and edits fail safely with actionable errors.

#### 2.3.1: Tool API Design and Registry Updates
- **Description:** Define new tool surfaces and align them with ToolRegistry expectations.
- **Tasks:**
  - Review existing file tools (`read_file`, `write_file`, `source_map`) for extension points.
  - Define schemas for `find_file`, `find_section`, and `edit_file` tools.
  - Update tool metadata (risk level, runner preference, network needs).
- **Files:**
  - `lsm/agents/tools/base.py`
  - `lsm/agents/tools/**`
- **success criteria:** Tool definitions are registered and discoverable with consistent schemas.

#### 2.3.2: Line-Hash Editing Engine
- **Description:** Implement a deterministic edit engine that replaces specific line spans safely.
- **Tasks:**
  - Define a line-hash format and collision rules.
  - Implement edit matching with descriptive failure output on mismatch.
  - Add `edit_file` tool to apply line-hash replacements.
- **Files:**
  - `lsm/agents/tools/edit_file.py`
  - `lsm/utils/file_graph.py`
- **success criteria:** Edits apply only when hashes match and failures include actionable diagnostics.

#### 2.3.3: Find and Section Location Tools
- **Description:** Provide fast file/section discovery using names, content, and structural graphs.
- **Tasks:**
  - Implement `find_file` with name and content search.
  - Implement `find_section` to resolve graph nodes by heading/function/class name.
  - Add graph-aware section filters (language, node type, depth).
- **Files:**
  - `lsm/agents/tools/find_file.py`
  - `lsm/agents/tools/find_section.py`
  - `lsm/utils/file_graph.py`
- **success criteria:** File and section discovery works in a single tool call for common queries.

#### 2.3.4: Read/Outline Enhancements
- **Description:** Improve read tools to provide outline and section-limited outputs.
- **Tasks:**
  - Extend `read_file` to accept section targets and max depth.
  - Extend `source_map` to return structural outlines derived from file graphs.
  - Add optional line-hash emission to read outputs.
- **Files:**
  - `lsm/agents/tools/read_file.py`
  - `lsm/agents/tools/source_map.py`
  - `lsm/utils/file_graph.py`
- **success criteria:** Read tools deliver section-only responses and structural outlines consistently.

#### 2.3.5: Tests and Documentation
- **Description:** Validate tool behavior and document new usage.
- **Tasks:**
  - Add tests for find/read/edit flows using fixtures.
  - Add performance benchmarks comparing line-hash editing vs baseline replace workflows.
  - Document tool schemas and examples in `docs/`.
- **Files:**
  - `tests/test_tools/`
  - `docs/`
- **success criteria:** Tooling works end-to-end with documented schemas and passing tests.

### 2.4: Execution Runners and Shell Tools
- **Description:** Add execution tools and sandbox validation for command runs.
- **Tasks:**
  - Complete Docker runner and add WSL2 runner.
  - Implement bash and PowerShell tools with allow/deny policies.
  - Update sandbox policy and add tests.
- **Files:**
  - `lsm/agents/tools/**`
  - `lsm/config/**`
  - `docs/`
  - `tests/**`
- **success criteria:** Runners execute within sandbox constraints and command tooling honors allow/deny config.

#### 2.4.1: Runner Policy and Config Updates
- **Description:** Align runner selection with sandbox policy and add configuration hooks for new runners.
- **Tasks:**
  - Audit current `ToolSandbox` runner selection rules.
  - Add configuration fields for WSL2 runner and command allow/deny lists.
  - Update runner selection to respect `execution_mode` and `force_docker` policy.
- **Files:**
  - `lsm/agents/tools/sandbox.py`
  - `lsm/config/**`
- **success criteria:** Runner selection is policy-driven and configurable.

#### 2.4.2: Docker Runner Completion
- **Description:** Finish Docker runner behavior and evaluate hot-load execution paths.
- **Tasks:**
  - Implement missing Docker runner features (volume mapping, environment scrubbing, timeout limits).
  - Evaluate hot-load strategy vs per-run container spawning.
  - Document runner constraints and failure modes.
- **Files:**
  - `lsm/agents/tools/docker_runner.py`
  - `lsm/agents/tools/env_scrubber.py`
  - `docs/`
- **success criteria:** Docker runner executes tool commands with enforced limits and predictable lifecycle behavior.

#### 2.4.3: WSL2 Runner Implementation
- **Description:** Add a Windows-hosted WSL2 runner for exec/network tools.
- **Tasks:**
  - Implement `WSL2Runner` with path translation and environment scrubbing.
  - Integrate runner availability checks in `ToolSandbox`.
  - Add logging and error handling for WSL2 invocation failures.
- **Files:**
  - `lsm/agents/tools/wsl2_runner.py`
  - `lsm/agents/tools/sandbox.py`
- **success criteria:** WSL2 runner executes commands in sandbox constraints when enabled.

#### 2.4.4: Bash and PowerShell Tools
- **Description:** Provide shell command tools with allow/deny constraints and path validation.
- **Tasks:**
  - Add `bash` tool for command execution with allow/deny config.
  - Add `powershell` tool for command execution with allow/deny config.
  - Implement path validation and sandbox enforcement for command arguments.
- **Files:**
  - `lsm/agents/tools/bash.py`
  - `lsm/agents/tools/powershell.py`
  - `lsm/agents/tools/sandbox.py`
- **success criteria:** Shell tools honor allow/deny lists and validate file paths before execution.

#### 2.4.5: Tests and Documentation
- **Description:** Validate runner and shell behavior with policy tests.
- **Tasks:**
  - Add unit tests for runner selection and command allow/deny behavior.
  - Add integration tests for Docker and WSL2 runners (when available).
  - Document config options and usage examples.
- **Files:**
  - `tests/test_agents_tools/`
  - `docs/`
- **success criteria:** Runner and shell tools are tested and documented with clear configuration examples.

### 2.5: Phase 2 Review and Changelog
- **Description:** Finalize Phase 2 with code review and documentation updates.
- **Tasks:**
  - Review for backwards compatibility and dead code.
  - Add tests for new tools and runners.
  - Update `docs/CHANGELOG.md` with agent/tooling changes.
- **Files:**
  - `docs/CHANGELOG.md`
  - `tests/**`
- **success criteria:** Phase 2 tool/agent changes are tested, reviewed, and documented.

## Phase 3: Agent Catalog and Meta-Agents

### 3.1: Core Agents
- **Description:** Implement base and specialized agents for v0.7.0.
- **Tasks:**
  - Extend agent registry and UI/shell lists.
  - Implement General, Librarian, Assistant, Coder, and Manuscript Editor agents.
  - Add tests and ensure standard artifacts are produced.
- **Files:**
  - `lsm/agents/**`
  - `lsm/agents/factory.py`
  - `lsm/ui/**`
- **success criteria:** Agents can be instantiated, run with tools, and produce summaries/artifacts.

#### 3.1.1: Agent Catalog and Registration
- **Description:** Extend the existing agent registry to include new agent types and themes.
- **Tasks:**
  - Add new agent names to `lsm/agents/factory.py` and module exports.
  - Update UI/shell agent lists to include new types.
  - Ensure agent metadata (description, default tools, risk posture) is exposed.
- **Files:**
  - `lsm/agents/factory.py`
  - `lsm/ui/shell/commands/agents.py`
  - `lsm/ui/tui/screens/agents.py`
- **success criteria:** New agents are discoverable in UI and shell commands.

#### 3.1.2: General Agent
- **Description:** Provide a general-purpose task agent that uses available tools responsibly.
- **Tasks:**
  - Define prompt and tool allowlist defaults.
  - Implement run outputs (summary + artifact list).
  - Add guardrails for permission and iteration limits.
- **Files:**
  - `lsm/agents/general.py`
  - `lsm/agents/harness.py`
- **success criteria:** General agent can execute tool-based tasks and emits standard artifacts.

#### 3.1.3: Librarian Agent
- **Description:** Explore embeddings DB and create idea graphs/metadata summaries.
- **Tasks:**
  - Define embeddings query workflow and output format.
  - Integrate graph output with `file_graph` and memory tools.
  - Emit `idea_graph.md` and supporting artifacts.
- **Files:**
  - `lsm/agents/librarian.py`
  - `lsm/agents/tools/query_embeddings.py`
- **success criteria:** Librarian agent produces a structured idea graph and citations.

#### 3.1.4: Assistant Agent
- **Description:** Summarize cross-agent activity and suggest memory updates.
- **Tasks:**
  - Parse `run_summary.json` artifacts for summaries.
  - Generate memory candidate suggestions with rationale.
  - Integrate with memory tools for promotion/rejection workflow.
- **Files:**
  - `lsm/agents/assistant.py`
  - `lsm/agents/memory/**`
- **success criteria:** Assistant agent produces consolidated summaries and actionable memory candidates.

#### 3.1.5: Coder Agent
- **Description:** Specialized agent for code editing tasks.
- **Tasks:**
  - Define code-editing workflow with find/read/edit tooling.
  - Emit code change summaries and touched file lists.
  - Enforce read/write scope based on sandbox rules.
- **Files:**
  - `lsm/agents/coder.py`
  - `lsm/agents/tools/**`
- **success criteria:** Coder agent can plan, edit, and summarize code changes reliably.

#### 3.1.6: Manuscript Editor Agent
- **Description:** Specialized agent for iteratively editing text documents.
- **Tasks:**
  - Define manuscript editing workflow (outline -> revisions -> final draft).
  - Use text graphing for section-level edits.
  - Emit revision logs and a final manuscript artifact.
- **Files:**
  - `lsm/agents/manuscript_editor.py`
  - `lsm/utils/file_graph.py`
- **success criteria:** Manuscript editor produces revised documents with traceable edits.

### 3.2: Communication Assistants
- **Description:** Implement email, calendar, and news assistants with explicit approval for side effects.
- **Tasks:**
  - Add Email Assistant for read/summary/draft with approval gating.
  - Add Calendar Assistant for read/schedule/edit with approval gating.
  - Add News Assistant for curated summaries and topic filtering.
- **Files:**
  - `lsm/agents/assistants/**`
  - `lsm/remote/**`
  - `lsm/agents/harness/**`
- **success criteria:** Assistants can read data, propose actions, and require explicit approval for changes.

### 3.3: Meta-Agents
- **Description:** Enable meta-agent orchestration with parallel sub-agent execution.
- **Tasks:**
  - Extend task graph planning for parallel execution.
  - Implement parallel sub-agent execution and artifact aggregation.
  - Add tests and documentation for meta-agent behavior.
- **Files:**
  - `lsm/agents/meta.py`
  - `lsm/agents/task_graph.py`
  - `lsm/agents/tools/**`
  - `docs/`
  - `tests/**`
- **success criteria:** Meta-agents can plan and execute sub-agents in parallel within sandbox constraints.

#### 3.3.1: Parallel Task Graph Planning
- **Description:** Extend task graph planning to represent parallelizable work.
- **Tasks:**
  - Update task graph models to allow parallel groups and dependency gates.
  - Add deterministic ordering for parallel execution plans.
  - Document graph serialization for meta-agent prompts.
- **Files:**
  - `lsm/agents/task_graph.py`
  - `lsm/agents/meta.py`
- **success criteria:** Meta plans include explicit parallel groups with deterministic ordering.

#### 3.3.2: Parallel Execution Engine
- **Description:** Execute sub-agents concurrently with resource limits and sandbox guarantees.
- **Tasks:**
  - Add concurrency controls to spawn/await workflows.
  - Enforce sandbox monotonicity on sub-agent execution.
  - Collect artifacts concurrently and merge results deterministically.
- **Files:**
  - `lsm/agents/meta.py`
  - `lsm/agents/tools/spawn_agent.py`
  - `lsm/agents/tools/await_agent.py`
  - `lsm/agents/tools/collect_artifacts.py`
- **success criteria:** Parallel runs complete safely with predictable artifact aggregation.

#### 3.3.3: General Meta-Agent
- **Description:** Provide a general meta-agent that orchestrates multiple agent types.
- **Tasks:**
  - Define meta-agent prompt and tool allowlist.
  - Implement sub-agent selection logic based on task types.
  - Emit final consolidated artifact (`final_result.md`).
- **Files:**
  - `lsm/agents/meta.py`
  - `lsm/agents/factory.py`
- **success criteria:** General meta-agent can coordinate multi-agent workflows end-to-end.

#### 3.3.4: Assistant Meta-Agent
- **Description:** Provide a meta-agent dedicated to assistant-style summaries and compliance checks.
- **Tasks:**
  - Define assistant-meta prompt for reviews, security checks, and memory proposals.
  - Add optional validation passes over sub-agent outputs.
  - Emit summary artifacts and action recommendations.
- **Files:**
  - `lsm/agents/meta.py`
  - `lsm/agents/assistant.py`
- **success criteria:** Assistant meta-agent produces structured reviews and action items.

#### 3.3.5: Tests and Documentation
- **Description:** Validate parallel meta-agent behavior and sandbox enforcement.
- **Tasks:**
  - Add tests for parallel plan execution and sandbox monotonicity.
  - Document meta-agent configuration and usage.
- **Files:**
  - `tests/test_agents_meta/`
  - `docs/`
- **success criteria:** Meta-agent parallel execution is tested and documented.

### 3.4: Phase 3 Review and Changelog
- **Description:** Finalize Phase 3 with code review and documentation updates.
- **Tasks:**
  - Review for backwards compatibility and dead code.
  - Add tests for new agents and meta-agents.
  - Update `docs/CHANGELOG.md` with agent catalog changes.
- **Files:**
  - `docs/CHANGELOG.md`
  - `tests/**`
- **success criteria:** Agent catalog and meta-agent features are tested, reviewed, and documented.

## Phase 4: AI Providers and Remote Sources

### 4.1: AI Provider Expansion and Tiering
- **Description:** Add OpenRouter support and model tier configuration.
- **Tasks:**
  - Implement OpenRouter provider.
  - Add tiered model selection (quick/normal/complex) and default mapping logic.
  - Ensure all tools/agents declare required tier in configuration.
- **Files:**
  - `lsm/providers/openrouter.py`
  - `lsm/config/**`
  - `docs/`
- **success criteria:** OpenRouter works end-to-end and tiered model mapping is enforced during config load.

### 4.2: Remote Provider Framework
- **Description:** Expand remote provider base and consolidate OAI-protocol providers.
- **Tasks:**
  - Implement generic RSS reader with caching.
  - Ensure OAI-protocol providers derive from `BaseOAIProvider`.
  - Add MCP host support and exposure to agents.
- **Files:**
  - `lsm/remote/**`
  - `lsm/agents/**`
  - `docs/`
- **success criteria:** RSS caching works, OAI providers share base, and MCP host is available to agents.

### 4.3: Remote Source Providers (Initial Set)
- **Description:** Add remote sources with clear config and data ingest paths.
- **Tasks:**
  - Implement structured provider groups (scholarly, cultural heritage, biomedical, email/calendar, news, social, financial).
  - Ensure every provider has concrete API integration (no placeholders).
  - Normalize outputs to `BaseRemoteProvider` schema and support ingest/download workflows.
- **Files:**
  - `lsm/remote/providers/**`
  - `lsm/ingest/**`
  - `lsm/remote/storage.py`
  - `lsm/config/**`
  - `docs/`
  - `example_config.json`
  - `.env.example`
- **success criteria:** Providers are configurable and can fetch/ingest data; stubs clearly document integration paths.
 - **success criteria:** Providers are configurable, fetch real data, and feed ingest workflows with normalized metadata.

#### 4.3.1: Scholarly Discovery Pipeline (OpenAlex -> Crossref -> Unpaywall -> CORE)
- **Description:** Implement a chained pipeline to discover and retrieve scholarly documents.
- **Tasks:**
  - Extend OpenAlex provider to return stable IDs and DOI metadata.
  - Extend Crossref provider to enrich metadata and resolve DOI/ISSN where needed.
  - Add Unpaywall provider for open-access link resolution.
  - Wire a pipeline that downloads full text when available and hands off to ingest.
- **Files:**
  - `lsm/remote/providers/openalex.py`
  - `lsm/remote/providers/crossref.py`
  - `lsm/remote/providers/unpaywall.py`
  - `lsm/remote/providers/core.py`
  - `lsm/remote/chain.py`
  - `lsm/ingest/**`
- **success criteria:** A DOI query yields a normalized metadata record plus full-text retrieval when available.

#### 4.3.2: Archives and Cultural Heritage Providers
- **Description:** Add providers for archives, museums, and cultural heritage datasets.
- **Tasks:**
  - Implement Archive.org provider for metadata + file retrieval.
  - Implement Perseus CTS API provider for text retrieval by CTS URNs.
  - Implement DPLA, Library of Congress, Smithsonian, Met, Rijksmuseum providers.
  - Implement IIIF search/presentation providers and Wikidata SPARQL provider.
- **Files:**
  - `lsm/remote/providers/archive_org.py`
  - `lsm/remote/providers/perseus_cts.py`
  - `lsm/remote/providers/dpla.py`
  - `lsm/remote/providers/loc.py`
  - `lsm/remote/providers/smithsonian.py`
  - `lsm/remote/providers/met.py`
  - `lsm/remote/providers/rijksmuseum.py`
  - `lsm/remote/providers/iiif.py`
  - `lsm/remote/providers/wikidata.py`
- **success criteria:** Each provider returns normalized results with stable IDs and source URLs.

#### 4.3.3: Biomedical and Academic Repositories
- **Description:** Add biomedical and discipline-specific sources.
- **Tasks:**
  - Implement PubMed and PubMed Central providers (E-utilities and OA full text when available).
  - Add SSRN provider for preprints and metadata.
  - Add PhilArchive and Project MUSE providers with structured metadata outputs.
- **Files:**
  - `lsm/remote/providers/pubmed.py`
  - `lsm/remote/providers/ssrn.py`
  - `lsm/remote/providers/philarchive.py`
  - `lsm/remote/providers/project_muse.py`
- **success criteria:** Queries return normalized metadata with open-access links when possible.

#### 4.3.4: Email and Calendar Providers
- **Description:** Add major email and calendar providers for assistant agents.
- **Tasks:**
  - Implement Gmail + Google Calendar providers via Google APIs.
  - Implement Microsoft Graph providers for Outlook mail and calendar.
  - Add IMAP/SMTP and CalDAV fallbacks for self-hosted providers.
  - Ensure strict approval gating before any send/update operation.
- **Files:**
  - `lsm/remote/providers/gmail.py`
  - `lsm/remote/providers/google_calendar.py`
  - `lsm/remote/providers/microsoft_graph_mail.py`
  - `lsm/remote/providers/microsoft_graph_calendar.py`
  - `lsm/remote/providers/imap.py`
  - `lsm/remote/providers/caldav.py`
- **success criteria:** Email/calendar assistants can read data and draft changes that require explicit user approval to apply.

#### 4.3.5: News Providers
- **Description:** Implement real news sources with API-backed retrieval.
- **Tasks:**
  - Implement NYTimes provider (Top Stories + Search APIs).
  - Implement Guardian or GDELT provider for global news coverage.
  - Implement NewsAPI (or equivalent) for topic aggregation.
  - Support RSS feeds for sources without APIs using the RSS reader.
- **Files:**
  - `lsm/remote/providers/nytimes.py`
  - `lsm/remote/providers/guardian.py`
  - `lsm/remote/providers/gdelt.py`
  - `lsm/remote/providers/newsapi.py`
  - `lsm/remote/providers/rss.py`
- **success criteria:** News queries return current articles with source, timestamp, and canonical URL.

#### 4.3.6: Social Media Providers
- **Description:** Add social platforms for public data gathering.
- **Tasks:**
  - Implement Twitter/X provider for search and recent posts.
  - Implement Reddit provider for subreddit/topic search.
  - Implement Mastodon provider for federated social search.
- **Files:**
  - `lsm/remote/providers/twitter.py`
  - `lsm/remote/providers/reddit.py`
  - `lsm/remote/providers/mastodon.py`
- **success criteria:** Social queries return normalized post metadata with source URLs.

#### 4.3.7: Financial and Government Data Providers
- **Description:** Provide market data and government datasets.
- **Tasks:**
  - Implement market data provider (Alpha Vantage or IEX Cloud).
  - Implement FRED provider for economic time series.
  - Implement SEC EDGAR provider for filings metadata and document retrieval.
  - Add Data.gov or Treasury provider for federal datasets.
- **Files:**
  - `lsm/remote/providers/alpha_vantage.py`
  - `lsm/remote/providers/iex_cloud.py`
  - `lsm/remote/providers/fred.py`
  - `lsm/remote/providers/sec_edgar.py`
  - `lsm/remote/providers/data_gov.py`
- **success criteria:** Financial queries return structured data with timestamps and source attribution.

#### 4.3.8: Web Search Alternatives
- **Description:** Add web search alternatives via AI provider search APIs.
- **Tasks:**
  - Evaluate AI-provider web search APIs as alternatives to Brave.
  - Implement at least one provider integration with consistent output schema.
  - Update configuration and documentation for selection.
- **Files:**
  - `lsm/remote/providers/web_search_ai.py`
  - `lsm/config/**`
  - `docs/`
- **success criteria:** Web search can be routed to a non-Brave provider via config.

#### 4.3.9: Tests and Documentation
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
- **success criteria:** Provider integrations are tested and documented with real config examples.

### 4.4: Phase 4 Review and Changelog
- **Description:** Finalize Phase 4 with code review and documentation updates.
- **Tasks:**
  - Review for backwards compatibility and dead code.
  - Add tests for providers and tiering logic.
  - Update `docs/CHANGELOG.md` with provider changes.
- **Files:**
  - `docs/CHANGELOG.md`
  - `tests/**`
- **success criteria:** AI provider and remote source changes are tested, reviewed, and documented.

## Phase 5: Release and Documentation Finalization

### 5.1: Version Bump and Documentation Audit
- **Description:** Update version metadata to v0.7.0 and confirm documentation reflects the release scope.
- **Tasks:**
  - Update version references to `0.7.0` in package metadata and runtime version fields.
  - Verify documentation reflects new agent/tool/remote provider capabilities and configuration.
  - Ensure examples (`example_config.json`, `.env.example`) include new config keys and API variables.
  - Update the TUI `WHATS_NEW` section with v0.7.0 highlights.
- **Files:**
  - `pyproject.toml`
  - `lsm/__init__.py`
  - `lsm/ui/tui/screens/help.py`
  - `tests/test_ui/tui/test_screens.py`
  - `docs/CHANGELOG.md`
  - `docs/**`
  - `.agents/docs/**`
  - `example_config.json`
  - `.env.example`
- **success criteria:** All version references are updated to 0.7.0 and documentation/config examples match the implemented changes.
