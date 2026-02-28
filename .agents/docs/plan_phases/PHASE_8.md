# Phase 8: RetrievalPipeline Core

**Status**: Pending

Implements the central `RetrievalPipeline` abstraction with the three-stage API
(`build_sources → synthesize_context → execute`), `QueryRequest`, `ContextPackage`,
`QueryResponse`, and supporting types. This is the foundation for all retrieval features.

Reference: [RESEARCH_PLAN.md §5.2, §5.3, §5.4](../RESEARCH_PLAN.md#52-unified-retrievalpipeline-abstraction)

---

## 8.1: Data Types — QueryRequest, ContextPackage, QueryResponse

**Description**: Define the core pipeline data types.

**Tasks**:
- Create `lsm/query/pipeline_types.py` (or `lsm/query/types.py`):
  - `FilterSet` dataclass: `path_contains`, `ext_allow`, `ext_deny`
  - `ScoreBreakdown` dataclass: `dense_score`, `dense_rank`, `sparse_score`,
    `sparse_rank`, `fused_score`, `rerank_score`, `temporal_boost`
  - `Citation` dataclass: `chunk_id`, `source_path`, `heading`, `page_number`,
    `url_or_doi`, `snippet`
  - `RetrievalTrace` dataclass: stages executed, timing, intermediate candidates,
    HyDE documents
  - `CostEntry` dataclass: tokens, cost
  - `RemoteSource` dataclass: provider, content, metadata
  - `QueryRequest` dataclass (per §5.3)
  - `ContextPackage` dataclass (per §5.4)
  - `QueryResponse` dataclass (per §5.4)
- Make `QueryRequest` cache/prompt fields explicit in the type definition:
  - `starting_prompt`, `conversation_id`, `prior_response_id`
- Make conversation-chain output fields explicit in `QueryResponse`:
  - `conversation_id`, `response_id`
- Extend existing `Candidate` in `lsm/query/session.py`:
  - Add `score_breakdown: Optional[ScoreBreakdown] = None`
  - Add `embedding: Optional[List[float]] = None` (for MMR diversity)
- Ensure all types are serializable (for agent tool responses)

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/pipeline_types.py` — new module with all types
- `lsm/query/session.py` — extend `Candidate`
- `tests/test_query/test_pipeline_types.py`:
  - Test: all dataclasses construct with defaults
  - Test: ScoreBreakdown fields are optional
  - Test: QueryRequest mode resolution (None → GROUNDED_MODE)
  - Test: QueryRequest/QueryResponse conversation fields serialize correctly
  - Test: serialization round-trip

**Success criteria**: All pipeline data types are defined and constructable. Types are
importable from `lsm.query`. Serialization works for agent tool transport.

---

## 8.2: RetrievalPipeline — Three-Stage API

**Description**: Implement the `RetrievalPipeline` class with `build_sources()`,
`synthesize_context()`, `execute()`, and `run()`.

**Tasks**:
- Create `lsm/query/pipeline.py`:
  - `RetrievalPipeline.__init__(db, embedder, config, llm_provider)`:
    - `db: BaseVectorDBProvider`
    - `embedder`: embedding function
    - `config: LSMConfig`
    - `llm_provider: BaseLLMProvider`
  - `build_sources(request: QueryRequest) -> ContextPackage`:
    - Resolve mode: `request.mode or GROUNDED_MODE`
    - Embed query using embedder
    - Dense recall via `db.query(embedding, top_k=k_dense)`
    - Apply metadata filters from `request.filters`
    - (Sparse recall and RRF fusion added in Phase 10)
    - Apply local reranking (migrate existing `rerank.py` logic)
    - Fetch remote sources if mode policy enables them
    - Build `ContextPackage` with candidates, traces, costs
  - `synthesize_context(package: ContextPackage) -> ContextPackage`:
    - Assign `[S1]`/`[S2]`/... labels to candidates
    - Build `source_labels` mapping
    - Format `context_block` string (migrate `context.py` logic)
    - Resolve `starting_prompt`:
      1. Explicit from `QueryRequest.starting_prompt` (highest priority)
      2. Session cache continuation (if `prior_response_id` set)
      3. Mode-derived default from `ModeConfig.synthesis_instructions`
    - Set `package.starting_prompt`, `package.context_block`, `package.source_labels`
  - `execute(package: ContextPackage) -> QueryResponse`:
    - Resolve synthesis LLM from resolved config + `ModeConfig` (no ad-hoc per-call
      selector field on `QueryRequest` — §9 Decision #7). The pipeline's `__init__`
      receives the synthesis provider; mode does not override provider selection.
    - Call `provider.send_message(input=package.context_block,
      instruction=package.starting_prompt, ...)`
    - Pass prompt/conversation cache parameters from package/config:
      - `previous_response_id=package.prior_response_id` when continuation applies
      - deterministic `prompt_cache_key` (provider/model/mode + conversation scope)
      - `prompt_cache_retention` when configured
    - On provider return, set `QueryResponse.response_id` from provider `last_response_id`
      and carry forward `QueryResponse.conversation_id`
    - Parse citations from response
    - Build `QueryResponse` with answer, package, citations, costs
  - `run(request: QueryRequest) -> QueryResponse`:
    - `build_sources → synthesize_context → execute`
- Migrate existing query logic from `api.py` into pipeline methods

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/pipeline.py` — RetrievalPipeline class
- `lsm/query/api.py` — refactor to use pipeline (retain as thin wrapper)
- `tests/test_query/test_pipeline.py`:
  - Test: `build_sources` returns ContextPackage with candidates
  - Test: `synthesize_context` assigns labels and formats context
  - Test: `execute` calls provider.send_message with correct params
  - Test: `execute` forwards `prior_response_id` and cache params correctly
  - Test: `execute` returns `response_id` from provider state
  - Test: `run` chains all three stages
  - Test: starting prompt resolution priority order
  - Test: conversation ID chaining (prior_response_id → response_id)

**Success criteria**: `RetrievalPipeline` implements the three-stage API. `run()` produces
a `QueryResponse` with answer and citations. Existing query functionality is preserved.

---

## 8.3: Session and TUI Integration

**Description**: Wire the `RetrievalPipeline` into the existing TUI/shell query path.
`SessionState` retains its role as artifact store but delegates query execution to the
pipeline.

**Tasks**:
- Update `lsm/query/api.py`:
  - Replace `_synthesize_answer()` with `pipeline.execute()`
  - Replace candidate retrieval with `pipeline.build_sources()`
  - Build `QueryRequest` from `SessionState` filters, model, mode, and conversation fields
    (`starting_prompt`, `conversation_id`, `prior_response_id`)
  - Map `QueryResponse` back to `SessionState` artifacts (last_candidates, last_answer, etc.)
  - Map `QueryResponse.response_id`/`conversation_id` back to `SessionState` for next turn
  - Define conversation-chain invalidation rules:
    - reset chain on model/provider/mode switch
    - reset chain when chat mode switches to `single`
- Update `SessionState`:
  - `conversation_id` and `prior_response_id` replace `llm_server_cache_ids` for
    the primary query flow
  - Remove `llm_server_cache_ids` — no backwards compatibility shims
- Update query-cache behavior for chat mode:
  - ensure cache hits preserve conversation-chain continuity
    (`conversation_id`/`prior_response_id` state remains valid after hit)
- Update TUI query screen to construct `QueryRequest` and display `QueryResponse`
- Add TUI retrieval trace view: display `ContextPackage.retrieval_trace` data
  (stages executed, timing, intermediate candidates, HyDE documents) in a debug
  panel or trace command. Traces are first-class artifacts per §7.3.
- Update shell/TUI query command handlers and settings wiring for new conversation cache
  fields and toggles

- Commit and push changes for this sub-phase.
**Files**:
- `lsm/query/api.py` — pipeline integration
- `lsm/query/session.py` — conversation state fields
- `lsm/ui/tui/screens/query.py` — QueryRequest construction
- `lsm/ui/helpers/commands/query.py` — query command integration
- `lsm/ui/tui/widgets/settings_query.py` — query cache/conversation settings wiring
- `lsm/ui/tui/state/settings_view_model.py` — settings apply/update wiring
- `tests/test_query/test_api.py` — integration tests
- `tests/test_query/test_chat_modes.py` — conversation/caching migration tests
- `tests/test_ui/tui/test_query_screen.py` — TUI tests
- `tests/test_ui/tui/test_query_commands.py` — command behavior tests
- `tests/test_ui/tui/test_settings_tabs.py` — settings persistence tests

**Success criteria**: TUI and shell query paths use the pipeline. Query results are
identical to pre-pipeline behavior. Conversation chaining works via `QueryRequest` /
`QueryResponse` IDs. Conversation invalidation rules are deterministic. Retrieval trace
is viewable in TUI.

---

## 8.4: Phase 8 Code Review and Changelog

**Tasks**:
- Review pipeline stage boundaries — each stage must be independently testable
- Review that `SessionState` artifacts are populated correctly from `QueryResponse`
- Review conversation state migration — no data loss during switchover
- Review prompt/conversation caching flow end-to-end:
  - `QueryRequest` fields propagate to `provider.send_message(...)` cache params
  - `QueryResponse.response_id` propagation is correct for next-turn chaining
- Review chat query-cache interaction — cache hits do not break conversation continuity
- Review performance — pipeline should not add measurable overhead vs. direct calls
- Run full test suite: `pytest tests/ -v`
- Update `docs/CHANGELOG.md`
- Update `.agents/docs/architecture/development/QUERY.md`
- Update `.agents/docs/architecture/packages/lsm.query.md`

- Commit and push changes for this sub-phase.
**Files**:
- `docs/CHANGELOG.md`
- `.agents/docs/architecture/development/QUERY.md`
- `.agents/docs/architecture/packages/lsm.query.md`

**Success criteria**: `pytest tests/ -v` passes. Pipeline is the single query code path.
Changelog and docs updated.

---
