# Query Modes Guide

Query modes define how LSM retrieves sources, which sources are allowed, and how
the answer is synthesized. Modes live in the config under `modes` and are
selected via `query.mode` or `/mode` in the query REPL.

## Built-In Modes

LSM ships with three built-in modes when `modes` is not defined:

| Mode | Synthesis Style | Local Sources | Remote Sources | Model Knowledge | Typical Use |
| --- | --- | --- | --- | --- | --- |
| `grounded` | `grounded` | enabled | disabled | disabled | Strict Q&A from local sources |
| `insight` | `insight` | enabled | disabled | enabled | Thematic synthesis across local docs |
| `hybrid` | `grounded` | enabled | enabled | enabled | Broad research with local + remote |

These defaults are defined in `lsm/config/models.py`.

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
      "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 10 },
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

- `min_relevance`: minimum relevance (1 - distance) to synthesize
- `k`: number of chunks to retrieve from Chroma
- `k_rerank`: number of chunks to keep after reranking

### Remote Source Policy

- `enabled`: enable remote sources for this mode
- `rank_strategy`: `weighted`, `sequential`, `interleaved` (reserved for future)
- `max_results`: max results per provider

### Model Knowledge Policy

- `enabled`: allow LLM to use its own training knowledge
- `require_label`: if true, model knowledge should be explicitly labeled

## Mode Selection

Modes are selected by:

- `query.mode` in config (default for all sessions)
- `/mode <name>` in the query REPL (session override)

The REPL shows mode details via `/mode`.

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
      "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
      "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 5 },
      "model_knowledge": { "enabled": true, "require_label": true }
    },
    "notes": { "enabled": true, "dir": "research_notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

## Mode Comparison Tips

- If you want strict citation answers, use `grounded`.
- If you want broader synthesis across a corpus, use `insight`.
- If you want web results displayed alongside local sources, use `hybrid`.

## Current Limitations

- `rank_strategy` is not yet applied when merging remote results.
- Remote results are displayed but not merged into the LLM context yet.
- `model_knowledge.require_label` is advisory in the current prompt.
