# Query Pipeline Architecture

This document describes how LSM answers queries from the local knowledge base.

## Goals

- Retrieve relevant chunks quickly.
- Filter and rerank candidates for quality.
- Synthesize answers with citations.
- Provide debuggable, repeatable results.

## High-Level Flow

```
question -> embed -> retrieve -> filter -> rerank -> gate -> synthesize -> format
```

## Core Components

### Query Embedding

- Implemented in `lsm/query/retrieval.py`.
- Uses the same sentence-transformer model as ingest.
- Normalizes embeddings for cosine similarity.

### Retrieval

- Uses ChromaDB vector similarity search.
- Retrieves `k` candidates (or `retrieve_k` if filters are active).
- Returns `Candidate` objects with text, metadata, and distance.

### Filtering

- Optional filters from config or REPL session:
  - `path_contains`
  - `ext_allow`
  - `ext_deny`

### Reranking

Local reranking (`lsm/query/rerank.py`) includes:

1. Deduplication by normalized text.
2. Lexical scoring with token overlap and phrase bonus.
3. Diversity enforcement (max chunks per file).

LLM reranking is optional and uses the active provider (`BaseLLMProvider`).

### Relevance Gating

- The best relevance score is computed as `1 - distance`.
- If relevance is below `min_relevance`, the pipeline returns a fallback answer.

### Synthesis

- Uses `build_context_block` to produce a source list with `[S#]` labels.
- Uses provider `synthesize()` for grounded or insight styles.
- Adds a note if no inline citations are present.

### Notes

- Last query artifacts are stored in `SessionState`.
- `/note` opens an editable Markdown note for saving.

### Remote Sources

- Enabled per mode using `source_policy.remote.enabled`.
- Remote results are fetched and displayed after the answer.
- Remote results are merged into the LLM context when enabled.

## Session Management

`SessionState` stores:

- Current filters (`path_contains`, `ext_allow`, `ext_deny`)
- Last candidates and chosen sources
- Last answer and remote sources
- Pinned chunk IDs for forced inclusion

## Query REPL

The query REPL (`lsm/query/repl.py`) supports:

- interactive questions
- model selection
- mode switching
- source inspection (`/show`, `/expand`, `/open`)
- session filters
- saving notes

See `docs/api-reference/REPL.md` for commands.

## Provider Interactions

The query pipeline uses two provider configurations:

- `llm.get_ranking_config()` for reranking
- `llm.get_query_config()` for synthesis

Per-feature overrides allow a cheaper model for reranking or tagging.
