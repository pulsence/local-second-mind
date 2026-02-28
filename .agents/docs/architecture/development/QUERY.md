# Query Pipeline Architecture

This document describes how LSM answers queries from the local knowledge base.

## Goals

- Retrieve relevant chunks quickly.
- Filter and rerank candidates for quality.
- Synthesize answers with citations.
- Provide debuggable, repeatable results.

## High-Level Flow

```
question → QueryRequest → RetrievalPipeline
  build_sources:      embed → retrieve → filter → rerank → remote
  synthesize_context:  label → context_block → prompt
  execute:             provider.send_message → citations → costs
→ QueryResponse → QueryResult
```

## RetrievalPipeline

The central pipeline abstraction in `lsm/query/pipeline.py` exposes three stages:

1. **build_sources(request)** → `ContextPackage`
   - Routes through the configured retrieval profile (see below)
   - Fetches remote sources when enabled (`context.fetch_remote_sources()`)
   - Records stage timings and profile name in `RetrievalTrace`

2. **synthesize_context(package)** → `ContextPackage`
   - Assigns `[S#]` source labels via `build_context_block()`
   - Resolves `starting_prompt` (explicit > mode default)
   - Populates `context_block` and `source_labels`

3. **execute(package)** → `QueryResponse`
   - Calls `provider.send_message()` with context and prompt
   - Parses `[S#]` citations from the answer
   - Tracks token costs; captures `response_id` for conversation chaining
   - Handles model_knowledge note and citation warnings

**run(request)** chains all three stages with a relevance gate (early exit
when best relevance < `min_relevance`).

## Pipeline Data Types

Defined in `lsm/query/pipeline_types.py`:

| Type | Purpose |
|------|---------|
| `QueryRequest` | Immutable request: question, mode, filters, k, conversation chain |
| `ContextPackage` | Mutable accumulator across stages: candidates, remote_sources, trace |
| `QueryResponse` | Final result: answer, citations, costs, package |
| `FilterSet` | path_contains, ext_allow, ext_deny |
| `ScoreBreakdown` | Per-candidate dense/sparse/fused/rerank scores |
| `Citation` | Structured citation: chunk_id, source_path, heading, snippet |
| `RetrievalTrace` | Stages executed, timings, retrieval_profile |
| `CostEntry` | Provider, model, tokens, cost |
| `RemoteSource` | Provider, title, url, snippet, score |
| `StageTimings` | Stage name + duration_ms |

## API Layer

`lsm/query/api.py` provides `query()` and `query_sync()` — thin wrappers that:

1. Check the query cache
2. Build a `QueryRequest` from `SessionState`
3. Create a `RetrievalPipeline` and call `pipeline.run()`
4. Map `QueryResponse` back to `SessionState` artifacts and `QueryResult`
5. Handle conversation chaining (chat mode) and auto-save

## Session Management

`SessionState` stores:

- Current filters (`path_contains`, `ext_allow`, `ext_deny`)
- Last candidates and chosen sources
- Last answer and remote sources
- Pinned chunk IDs for forced inclusion
- Conversation chain state (`conversation_id`, `prior_response_id`)
- Last retrieval trace for diagnostics

## Retrieval Profiles

The pipeline supports five retrieval profiles configured via `retrieval_profile`:

| Profile | Stages | Status |
|---------|--------|--------|
| `dense_only` | Dense vector recall only | Implemented |
| `hybrid_rrf` | Dense + Sparse (BM25/FTS5) + RRF fusion | Implemented |
| `llm_rerank` | Dense + LLM reranking | Implemented |
| `hyde_hybrid` | HyDE embedding + hybrid_rrf | Placeholder (Phase 11) |
| `dense_cross_rerank` | Dense + cross-encoder reranking | Placeholder (Phase 11) |

Profile routing lives in `build_sources()` with `_profile_dense_only()`,
`_profile_hybrid_rrf()`, and `_profile_llm_rerank()` methods.

### RRF Fusion (`lsm/query/stages/rrf_fusion.py`)

Implements Reciprocal Rank Fusion (Cormack et al. 2009):

```
fused_score(d) = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)
```

- Default k=60, dense_weight=0.7, sparse_weight=0.3
- Candidates missing from one list get default rank = len(list) + 1
- Preserves original `ScoreBreakdown` fields from upstream stages

### Retrieval Stages

- `lsm/query/stages/dense_recall.py` — Dense vector similarity via `db.query()`.
  Populates `dense_score` and `dense_rank` on `ScoreBreakdown`.
- `lsm/query/stages/sparse_recall.py` — BM25 full-text via `db.fts_query()`.
  Populates `sparse_score` and `sparse_rank` on `ScoreBreakdown`.
- `lsm/query/stages/rrf_fusion.py` — Merges dense + sparse via RRF.
  Populates `fused_score` and preserves upstream scores.

### Graceful Degradation

`_profile_hybrid_rrf()` probes FTS availability before executing sparse recall.
If FTS5 is not available, it falls back to `_profile_dense_only()` with a warning log.

## Core Components

### Retrieval

- Implemented in `lsm/query/retrieval.py` and `lsm/query/planning.py`.
- Uses sentence-transformer embeddings with sqlite-vec similarity search.
- Retrieves `k` candidates with optional `is_current = true` filtering.
- Returns `Candidate` objects with text, metadata, distance, and optional `score_breakdown`.

### Filtering

- Optional filters from config or TUI session:
  - `path_contains`
  - `ext_allow`
  - `ext_deny`

### Reranking

Local reranking (`lsm/query/rerank.py`) includes:

1. Deduplication by normalized text.
2. Lexical scoring with token overlap and phrase bonus.
3. Diversity enforcement (max chunks per file).

LLM reranking (`lsm/query/stages/llm_rerank.py`) is optional and uses the
ranking service provider.

### Relevance Gating

- The best relevance score is computed as `1 - distance`.
- If relevance is below `min_relevance`, the pipeline returns a fallback answer.

### Synthesis

- Uses `build_context_block` to produce a source list with `[S#]` labels.
- Uses `provider.send_message()` with mode-specific instructions from `lsm/query/prompts.py`.
- Adds a note if no inline citations are present.

### Remote Sources

- Enabled per mode using `remote_policy.enabled`.
- Remote results are fetched and merged into the LLM context when enabled.

## Query Commands (TUI)

The Query tab uses shared command handlers to support:

- interactive questions
- model selection
- mode switching
- source inspection (`/show`, `/expand`, `/open`)
- session filters
- saving notes

See `.agents/docs/architecture/api-reference/REPL.md` for commands.

## Provider Interactions

The query pipeline uses two provider configurations:

- `llm.resolve_service("ranking")` for reranking
- `llm.resolve_service("query")` for synthesis

Per-feature overrides allow a cheaper model for reranking or tagging.
