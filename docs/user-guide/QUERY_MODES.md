# Query Modes Guide

Query modes define how LSM retrieves sources, which sources are allowed, and how
the answer is synthesized. Modes live in the config under `modes` and are
selected via `query.mode` or `/mode` in the TUI Query tab.

## Built-In Modes

LSM ships with three built-in modes when `modes` is not defined:

| Mode | Synthesis Style | Local Sources | Remote Sources | Model Knowledge | Typical Use |
| --- | --- | --- | --- | --- | --- |
| `grounded` | `grounded` | enabled | disabled | disabled | Strict Q&A from local sources |
| `insight` | `insight` | enabled | disabled | enabled | Thematic synthesis across local docs |
| `hybrid` | `grounded` | enabled | enabled | enabled | Broad research with local + remote |

These defaults are defined in `lsm/config/models/`.

## What a Mode Controls

A mode has three sub-systems:

1. `synthesis_style` (how answers are written)
2. `source_policy` (which sources can be used and how)
3. `notes` (how query sessions are saved)

### Synthesis Styles

- `grounded`: Answers must cite sources and avoid unsupported claims.
- `insight`: Summarize patterns, tensions, and themes across sources.

### Source Policy

`source_policy` has three parts:

- `local`: how many local chunks to retrieve and rerank
- `remote`: whether remote sources are allowed and how many results to fetch
- `model_knowledge`: whether the model can use its own training knowledge

### Notes Policy

`notes` controls whether a query session can be saved and where it is stored.

## Mode Configuration Reference

```json
"modes": [
  {
    "name": "research",
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "min_relevance": 0.20, "k": 15, "k_rerank": 8 },
      "remote": {
        "enabled": true,
        "rank_strategy": "weighted",
        "max_results": 10,
        "remote_providers": ["brave", "wikipedia", "arxiv"]
      },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": {
      "enabled": true,
      "dir": "research_notes",
      "template": "default",
      "filename_format": "timestamp"
    }
  }
]
```

### Local Source Policy

- `enabled`: enable local sources for this mode
- `min_relevance`: minimum relevance (1 - distance) to synthesize
- `k`: number of chunks to retrieve from Chroma
- `k_rerank`: number of chunks to keep after reranking

### Remote Source Policy

- `enabled`: enable remote sources for this mode
- `rank_strategy`: `weighted`, `sequential`, `interleaved` (reserved for future)
- `max_results`: max results per provider
- `remote_providers`: optional list of `remote_providers[].name` values to use for this mode

### Model Knowledge Policy

- `enabled`: allow LLM to use its own training knowledge
- `require_label`: if true, model knowledge should be explicitly labeled

## Mode Selection

Modes are selected by:

- `query.mode` in config (default for all sessions)
- `/mode <name>` in the TUI Query tab (session override)

The TUI shows mode details via `/mode`.

## Chat Mode and Caching

Query mode selection (`/mode <name>`) is separate from query response mode:

- `query.chat_mode = "single"`: stateless turns.
- `query.chat_mode = "chat"`: conversation history is tracked and sent as follow-up context.

Caching options for query/chat:

- `query.enable_query_cache`: local in-memory TTL/LRU result cache.
- `query.enable_llm_server_cache`: provider-side server cache/session reuse for follow-up chat turns (enabled by default).

Live toggle in TUI:

- `/mode set llm_cache on`
- `/mode set llm_cache off`

`/mode set` also supports:

- `model_knowledge`
- `remote`
- `notes`

## Custom Mode Examples

### Example: Strict Local-Only Q&A

```json
"modes": [
  {
    "name": "local_only",
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "min_relevance": 0.30, "k": 10, "k_rerank": 5 },
      "remote": { "enabled": false },
      "model_knowledge": { "enabled": false }
    },
    "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

### Example: Insight-Only Analysis

```json
"modes": [
  {
    "name": "themes",
    "synthesis_style": "insight",
    "source_policy": {
      "local": { "min_relevance": 0.20, "k": 18, "k_rerank": 10 },
      "remote": { "enabled": false },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "analysis_notes", "template": "default", "filename_format": "query_slug" }
  }
]
```

### Example: Research With Remote Sources

```json
"modes": [
  {
    "name": "research",
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "enabled": true, "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
      "remote": {
        "enabled": true,
        "rank_strategy": "weighted",
        "max_results": 5,
        "remote_providers": ["arxiv", "wikipedia"]
      },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "research_notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

### Example: Remote-Only Mode

```json
"modes": [
  {
    "name": "remote_only",
    "synthesis_style": "insight",
    "source_policy": {
      "local": { "enabled": false },
      "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 5 },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

## Mode Comparison Tips

- If you want strict citation answers, use `grounded`.
- If you want broader synthesis across a corpus, use `insight`.
- If you want web results displayed alongside local sources, use `hybrid`.

## Current Limitations

- `rank_strategy` is not yet applied when merging remote results.
- `model_knowledge.require_label` is advisory in the current prompt.
