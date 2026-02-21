# Phase 7: AI Providers & Protocol Infrastructure

**Why seventh:** OpenRouter and protocol work (OAI consolidation, MCP, RSS) must be in place before adding remote source providers in Phase 8.

**Depends on:** Phase 1.3 (tiered config), Phase 3.3 (function calling standardization)

| Task | Description | Depends On |
|------|-------------|------------|
| 7.1 | OpenRouter provider | 1.3 |
| 7.2 | OAI-protocol base class consolidation | None |
| 7.3 | MCP host support | 3.3 |
| 7.4 | Generic RSS reader with caching | None |

## 7.1: OpenRouter Provider
- **Description:** Add OpenRouter as an LLM provider, routing to any model available via OpenRouter's API.
- **Tasks:**
  - Implement `OpenRouterProvider` extending base LLM provider.
  - Support model selection via `provider_name: "openrouter"` in config with `model` field for specific model routing.
  - Handle OpenRouter-specific features: model fallbacks, prompt caching headers, usage tracking.
  - Add to `.env.example`: `OPENROUTER_API_KEY`.
  - Write tests for OpenRouter provider instantiation, model selection via config, fallback behavior, usage tracking, and integration with the provider registry (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_config/ tests/test_providers/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/providers/openrouter.py`
  - `lsm/config/models/llm.py`
  - `.env.example`
- **Success criteria:** OpenRouter provider works end-to-end for query and agent workflows.

## 7.2: OAI-Protocol Base Class Consolidation
- **Description:** Ensure all remote providers using OAI-PMH protocol derive from a shared `BaseOAIProvider` to reduce code duplication.
- **Tasks:**
  - Audit existing OAI-PMH providers: `arxiv.py`, `oai_pmh.py` — identify shared logic (record parsing, resumption tokens, metadata format mapping).
  - Extract shared OAI logic into `BaseOAIProvider` base class.
  - Refactor existing providers to inherit from `BaseOAIProvider`.
  - New OAI-based providers in Phase 8 must use this base.
  - Write tests for `BaseOAIProvider` shared logic (record parsing, resumption tokens, metadata mapping), and verify refactored providers maintain identical behavior (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_providers/remote/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/remote/providers/base_oai.py`
  - `lsm/remote/providers/arxiv.py`
  - `lsm/remote/providers/oai_pmh.py`
- **Success criteria:** OAI providers share common harvesting logic. Adding a new OAI repository requires only config, not new parsing code.

## 7.3: MCP Host Support
- **Description:** Enable Local Second Mind to act as an MCP (Model Context Protocol) host, exposing MCP tools to agents (and later to query workflows).
- **Tasks:**
  - Implement MCP host client that can connect to MCP servers.
  - Register MCP-provided tools in the `ToolRegistry` so agents can discover and use them.
  - Add configuration as a **global config** field: `mcp_servers: [{ name, command, args, env }]` under the `"global"` section of config. This is global (not agent-specific) because MCP servers will also be usable from query workflows in a future version.
  - Ensure MCP tool execution respects sandbox constraints when used by agents.
  - Wire MCP server lifecycle management: start servers on demand, restart on failure, shutdown on application exit.
  - Write tests for MCP host client connection, tool registration in ToolRegistry, config field validation, sandbox constraint enforcement on MCP tools, and server lifecycle management (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_agents/ tests/test_config/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/agents/tools/mcp_host.py`
  - `lsm/config/models/global_config.py`
  - `lsm/agents/tools/__init__.py`
- **Success criteria:** Agents can discover and invoke tools provided by external MCP servers. MCP tools appear in `ToolRegistry.list_tools()`. MCP config is in global section.

## 7.4: Generic RSS Reader
- **Description:** Implement a reusable RSS/Atom feed reader with caching for use by news providers and general feed consumption.
- **Tasks:**
  - Parse RSS 2.0 and Atom feeds.
  - Cache past reads with configurable TTL. Track seen items to download only new entries.
  - Normalize feed items to `RemoteResult` schema.
  - Expose as both a standalone remote provider (`rss` type) and a utility for other providers.
  - Write tests for RSS 2.0 and Atom feed parsing, cache TTL behavior, seen-item tracking (only new entries returned), and `RemoteResult` normalization (TDD: write tests before implementation).
  - Run the relevant test suite (`pytest tests/test_providers/remote/`) and verify all new and existing tests pass.
- **Files:**
  - `lsm/remote/providers/rss.py`
  - `lsm/remote/storage.py`
- **Success criteria:** RSS provider fetches feeds, caches results, and returns normalized items. Subsequent fetches return only new items.
