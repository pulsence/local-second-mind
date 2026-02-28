# Query Modes Guide

Query modes define how LSM retrieves sources and synthesizes answers. Modes are
configured under `modes` and selected with `query.mode` (or `/mode` in the TUI).

## Built-In Modes

When `modes` is omitted, LSM loads three built-in presets:

| Mode | retrieval_profile | Local policy | Remote policy | Model knowledge | synthesis_style |
| --- | --- | --- | --- | --- | --- |
| `grounded` | `hybrid_rrf` | enabled, `k=12`, `min_relevance=0.25` | disabled | disabled | `grounded` |
| `insight` | `hybrid_rrf` | enabled, `k=8`, `min_relevance=0.0` | enabled, `max_results=5` | enabled | `insight` |
| `hybrid` | `hybrid_rrf` | enabled, `k=12`, `min_relevance=0.15` | enabled, `max_results=5` | enabled | `grounded` |

## Mode Data Model

Each mode is a `ModeConfig` with these fields:

- `retrieval_profile`: retrieval strategy identifier (current default: `hybrid_rrf`)
- `synthesis_style`: answer style (`grounded` or `insight`)
- `synthesis_instructions`: full synthesis instruction prompt (user-overridable)
- `local_policy`: local retrieval policy (`enabled`, `k`, `min_relevance`)
- `remote_policy`: remote policy (`enabled`, `rank_strategy`, `max_results`, `remote_providers`)
- `model_knowledge_policy`: whether model priors are allowed (`enabled`, `require_label`)
- `notes`: note-writing behavior for the mode
- `chats`: optional per-mode transcript overrides (`auto_save`, `dir`)

Legacy note: older configs that used `source_policy` are still accepted by the
loader, but new configs should use `local_policy` / `remote_policy` /
`model_knowledge_policy`.

## Mode Configuration Reference

```json
"modes": [
  {
    "name": "research",
    "retrieval_profile": "hybrid_rrf",
    "synthesis_style": "grounded",
    "synthesis_instructions": "Answer using the provided context. Cite sources with [S#].",
    "local_policy": { "enabled": true, "k": 15, "min_relevance": 0.20 },
    "remote_policy": {
      "enabled": true,
      "rank_strategy": "weighted",
      "max_results": 10,
      "remote_providers": ["brave", "wikipedia", "arxiv"]
    },
    "model_knowledge_policy": { "enabled": true, "require_label": true },
    "notes": {
      "enabled": true,
      "dir": "research_notes",
      "template": "default",
      "filename_format": "timestamp"
    },
    "chats": {
      "auto_save": true,
      "dir": "Chats/Research"
    }
  }
]
```

## Policy Field Reference

### local_policy

- `enabled`: include local chunk retrieval
- `k`: final local chunk budget used for synthesis context
- `min_relevance`: minimum local relevance threshold

### remote_policy

- `enabled`: include remote sources
- `rank_strategy`: remote merge strategy (`weighted`, `sequential`, `interleaved`)
- `max_results`: max results fetched per remote source
- `remote_providers`: optional list of provider names (or weighted refs) for this mode

### model_knowledge_policy

- `enabled`: allow use of model training knowledge
- `require_label`: require explicit labeling when model knowledge is used

## Mode Selection

Modes are selected by:

- `query.mode` in config (default)
- `/mode <name>` in the TUI Query tab (session override)

`LSMConfig.validate()` ensures the selected mode exists, otherwise it falls back
to `grounded` (or first available mode).

## Chat Mode and Caching

Mode selection is separate from chat behavior:

- `query.chat_mode = "single"`: stateless turns
- `query.chat_mode = "chat"`: conversation history retained

Caching controls:

- `query.enable_query_cache`: local in-memory result cache
- `query.enable_llm_server_cache`: provider-side response-chain caching

TUI toggles:

- `/mode set llm_cache on`
- `/mode set llm_cache off`

## Custom Mode Examples

### Strict local-only answers

```json
"modes": [
  {
    "name": "local_only",
    "retrieval_profile": "hybrid_rrf",
    "synthesis_style": "grounded",
    "local_policy": { "enabled": true, "k": 10, "min_relevance": 0.30 },
    "remote_policy": { "enabled": false },
    "model_knowledge_policy": { "enabled": false },
    "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
  }
]
```

### Insight-oriented synthesis

```json
"modes": [
  {
    "name": "themes",
    "retrieval_profile": "hybrid_rrf",
    "synthesis_style": "insight",
    "local_policy": { "enabled": true, "k": 18, "min_relevance": 0.0 },
    "remote_policy": { "enabled": false },
    "model_knowledge_policy": { "enabled": true, "require_label": true }
  }
]
```

### Local + remote research

```json
"modes": [
  {
    "name": "research",
    "retrieval_profile": "hybrid_rrf",
    "synthesis_style": "grounded",
    "local_policy": { "enabled": true, "k": 12, "min_relevance": 0.15 },
    "remote_policy": {
      "enabled": true,
      "rank_strategy": "weighted",
      "max_results": 5,
      "remote_providers": ["arxiv", "wikipedia"]
    },
    "model_knowledge_policy": { "enabled": true, "require_label": true }
  }
]
```
