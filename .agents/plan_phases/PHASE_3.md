# Phase 3: query_knowledge_base Tool

**Plan**: [PLAN.md](../docs/PLAN.md)
**Version**: 0.7.1
**Status**: Complete

Implements the `query_knowledge_base` tool that wraps `query_sync()` and calls the full query
pipeline. Removes and deletes `query_embeddings`.

Reference: [RESEARCH_PLAN.md §3](../docs/RESEARCH_PLAN.md#3-knowledge-base-access-query_knowledge_base-tool)

---

## 3.1: QueryKnowledgeBaseTool Implementation

**Description**: Create `QueryKnowledgeBaseTool`, a built-in agent tool that calls the full
`query_sync()` pipeline and returns a JSON-serialized result.

**Tasks**:
- Create `lsm/agents/tools/query_knowledge_base.py`:
  - Class `QueryKnowledgeBaseTool(BaseTool)`
  - `name = "query_knowledge_base"`
  - `description`: describe that it queries the knowledge base using the full pipeline including
    reranking and LLM synthesis; returns a grounded answer with sources
  - `risk_level = "read_only"`
  - `input_schema`: `query` (string, required), optional `top_k` (int), optional `filters`
    (object)
  - `__init__(self, config: LSMConfig, embedder, collection)`: store injected dependencies
  - `execute(self, args: dict) -> str`:
    - Validate `query`; raise `ValueError` if missing or blank
    - Construct a fresh `SessionState` from config defaults on each call
    - Call `query_sync(question=query, config=self.config, state=state, embedder=self.embedder,
      collection=self.collection)`
    - Serialize `QueryResult` to JSON: include `answer`, `sources_display`, and top candidate
      fields (`id`, `score`, `text` truncated to `max_chars`)
    - Return the JSON string
- `query_sync()` is the sync wrapper around the async `query()` — verify it exists in
  `lsm/query/api.py` (or add a thin sync wrapper if needed)
- Run relevant tests: `pytest tests/agents/tools/test_query_knowledge_base.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/tools/query_knowledge_base.py` — new file
- `tests/agents/tools/test_query_knowledge_base.py` — new:
  - Test: `execute()` calls `query_sync()` with the injected config, embedder, collection
  - Test: a fresh `SessionState` is constructed on each call (state from call N is not
    present in call N+1)
  - Test: result JSON contains `answer` and `sources_display` keys
  - Test: missing `query` raises `ValueError`
  - Test: blank `query` (after strip) raises `ValueError`
  - Test: optional `top_k` is forwarded to the pipeline
  - Test: optional `filters` are forwarded if provided

**Success criteria**: `QueryKnowledgeBaseTool` runs end-to-end against a mocked `query_sync()`.
JSON output is well-formed. Input validation is tested. Fresh `SessionState` per call is verified.

---

## 3.2: Tool Registry — Add New Tool, Remove query_embeddings

**Description**: Register `query_knowledge_base` in the default tool registry and delete
`query_embeddings` entirely.

**Tasks**:
- In `create_default_tool_registry()` (wherever it lives — likely `lsm/agents/tools/__init__.py`
  or `lsm/agents/registry.py`):
  - Add `QueryKnowledgeBaseTool(config=config, embedder=embedder, collection=collection)` to
    the returned registry
  - Remove `QueryEmbeddingsTool` registration
  - Ensure `config`, `embedder`, and `collection` are available at registration time (they are
    already passed for `query_embeddings` — just use the same pattern)
- Delete `lsm/agents/tools/query_embeddings.py`
- Remove `query_embeddings` from all agent `tool_allowlist` definitions across all agents:
  `ResearchAgent`, `SynthesisAgent`, `CuratorAgent`, `WritingAgent`, and any others referencing it
- Remove any imports of `QueryEmbeddingsTool` in `__init__.py` or elsewhere
- Run relevant tests: `pytest tests/agents/tools/test_registry.py -v`
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `lsm/agents/tools/__init__.py` (or registry file) — register new tool, deregister old
- `lsm/agents/tools/query_embeddings.py` — DELETE
- `lsm/agents/academic/research.py` — remove `query_embeddings` from `tool_allowlist`
- `lsm/agents/productivity/writing.py` — remove `query_embeddings` from `tool_allowlist`
- Any other agent files referencing `query_embeddings` in their `tool_allowlist`
- `tests/agents/tools/test_registry.py` — new or extended:
  - Test: `create_default_tool_registry()` contains `query_knowledge_base`
  - Test: `create_default_tool_registry()` does NOT contain `query_embeddings`
  - Test: `lsm/agents/tools/query_embeddings.py` does not exist (file-system assertion)

**Success criteria**: `query_knowledge_base` is in the default registry. `query_embeddings` is
absent from the registry and deleted from disk. No import errors across the codebase.

---

## 3.3: Phase 3 Code Review and Changelog

**Tasks**:
- Review `QueryKnowledgeBaseTool`: verify `SessionState` is constructed fresh per call and not
  shared; verify JSON output schema is stable
- Verify no remaining references to `query_embeddings` anywhere in the codebase (grep for
  `query_embeddings` — result should be empty except in tests that assert its absence)
- Review registry changes: verify `create_default_tool_registry()` signature is unchanged;
  verify dependency injection pattern is consistent with existing tools
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md` with Phase 3 changes
- Update `.agents/docs/architecture/packages/lsm.agents.md`: document `query_knowledge_base`
  tool; note `query_embeddings` removal
- Commit and push using the format in `COMMIT_MESSAGE.md`

**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/packages/lsm.agents.md`

**Success criteria**: `pytest tests/ -v` passes. `query_embeddings.py` is deleted. No broken
imports anywhere.
