# Phase 6: Prompt Ownership Migration and ModeConfig

**Status**: Pending

Moves domain-specific prompt templates from `providers/helpers.py` to the pipeline stages
and config objects that own the domain logic. Updates `ModeConfig` with `synthesis_instructions`
and `retrieval_profile`.

Reference: [RESEARCH_PLAN.md §5.11, §6.6](../RESEARCH_PLAN.md#511-mode-as-composition-preset)

---

## 6.1: ModeConfig Update

**Description**: Update `ModeConfig` to the v0.8.0 structure with `retrieval_profile`,
`synthesis_instructions`, and restructured policies.

**Tasks**:
- Update `lsm/config/models/modes.py`:
  - Add `retrieval_profile: str = "hybrid_rrf"` field
  - Add `synthesis_instructions: str` field (defaults to `SYNTHESIZE_GROUNDED_INSTRUCTIONS`)
  - Remove `k_rerank` from `LocalSourcePolicy` (replaced by `k` as post-rerank count)
  - Verify `synthesis_style`, `local_policy`, `remote_policy`, `model_knowledge_policy`
    fields match §5.11
  - Add `ModeChatsConfig` if not already present
- Define built-in mode objects:
  ```python
  GROUNDED_MODE = ModeConfig(
      retrieval_profile="hybrid_rrf",
      synthesis_style="grounded",
      synthesis_instructions=SYNTHESIZE_GROUNDED_INSTRUCTIONS,
      local_policy=LocalSourcePolicy(k=12, min_relevance=0.25, enabled=True),
      remote_policy=RemoteSourcePolicy(enabled=False),
      model_knowledge_policy=ModelKnowledgePolicy(enabled=False),
  )
  INSIGHT_MODE = ModeConfig(...)  # per §5.11 table
  HYBRID_MODE = ModeConfig(...)   # per §5.11 table
  BUILT_IN_MODES: Dict[str, ModeConfig] = {...}
  ```
- Move `SYNTHESIZE_GROUNDED_INSTRUCTIONS` and `SYNTHESIZE_INSIGHT_INSTRUCTIONS` from
  `providers/helpers.py` to a new `lsm/query/prompts.py` (or inline in modes.py)
- Update mode resolution in query path to use `BUILT_IN_MODES` lookup
- Update config loader to parse new ModeConfig fields from config files
- Update all tests that reference old ModeConfig structure

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/config/models/modes.py` — ModeConfig restructure
- `lsm/query/prompts.py` — synthesis prompt templates (new file, or in modes.py)
- `lsm/config/loader.py` — mode config parsing
- `lsm/query/api.py` — mode resolution update
- `tests/test_config/test_modes.py` — updated tests
- `tests/test_query/test_modes.py` — updated tests

**Success criteria**: `ModeConfig` has `retrieval_profile` and `synthesis_instructions`.
Built-in modes are defined with correct defaults. Mode resolution via name works.

---

## 6.2: Rerank and Tag Prompt Migration

**Description**: Move rerank and tag generation prompts, utilities, and parsing from
`providers/helpers.py` to their owning modules.

**Tasks**:
- Create `lsm/query/stages/` directory (prepare for pipeline stages)
- Create `lsm/query/stages/llm_rerank.py`:
  - Move `RERANK_INSTRUCTIONS` from `providers/helpers.py`
  - Move `RERANK_JSON_SCHEMA` from `providers/helpers.py`
  - Move `prepare_candidates_for_rerank()` from `providers/helpers.py`
  - Move `parse_ranking_response()` from `providers/helpers.py`
  - Create `llm_rerank(candidates, query, provider, ...)` function that:
    - Constructs the rerank prompt
    - Calls `provider.send_message(instruction=rerank_prompt, input=payload, ...)`
    - Parses the response
    - Returns reranked candidates
- Update `lsm/ingest/tagging.py`:
  - Move `TAG_GENERATION_TEMPLATE` from `providers/helpers.py`
  - Move `TAGS_JSON_SCHEMA` from `providers/helpers.py`
  - Move `get_tag_instructions()` from `providers/helpers.py`
  - Update to call `provider.send_message(instruction=tag_prompt, input=text, ...)`
    instead of `provider.generate_tags()`
- Move `format_user_content()` from `providers/helpers.py` to `lsm/query/context.py`
- Move `generate_fallback_answer()` from `providers/helpers.py` to `lsm/query/` module
- Clean `providers/helpers.py`: only `parse_json_payload()` and `UnsupportedParamTracker`
  should remain
- Update all imports across the codebase

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/stages/__init__.py` — new package
- `lsm/query/stages/llm_rerank.py` — rerank prompt and logic
- `lsm/ingest/tagging.py` — tag prompt and logic
- `lsm/query/context.py` — `format_user_content()`
- `lsm/providers/helpers.py` — remove migrated assets
- `lsm/query/rerank.py` — update imports
- `tests/test_query/test_llm_rerank.py` — rerank stage tests
- `tests/test_ingest/test_tagging.py` — tagging tests
- `tests/test_providers/test_helpers.py` — updated (reduced scope)

**Success criteria**: `providers/helpers.py` contains only generic utilities. All domain
prompts live in the module that owns the domain logic. Rerank and tagging work correctly
via `send_message()`. All tests pass.

---

## 6.3: Phase 6 Code Review and Changelog

**Tasks**:
- Review that `providers/helpers.py` contains no domain-specific prompts or logic
- Review import graph: no circular dependencies between `providers/` and `query/stages/`
- Review ModeConfig defaults match §5.11 table exactly
- Review `synthesis_instructions` is user-overridable via config
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `docs/user-guide/QUERY_MODES.md` — document new ModeConfig fields
- Update `.agents/docs/architecture/development/MODES.md`
- Update `.agents/docs/architecture/development/PROVIDERS.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `docs/user-guide/QUERY_MODES.md`
- `.agents/docs/architecture/development/MODES.md`
- `.agents/docs/architecture/development/PROVIDERS.md`

**Success criteria**: `pytest tests/ -v` passes. Clean separation between providers and
domain logic. Changelog and docs updated.

---
