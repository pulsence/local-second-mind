# Configuration Guide

This guide covers every configuration option in Local Second Mind (LSM), how
configuration is loaded, and common setup patterns.

LSM uses a single configuration file in JSON or YAML format. Paths can be
relative or absolute. Relative paths are resolved relative to the config file
location.

## Supported Formats

- JSON: `config.json`
- YAML: `config.yaml` or `config.yml`

The config loader also reads environment variables from `.env` in the current
working directory via `python-dotenv`.

## Loading Order

1. `.env` is loaded into the environment.
2. The config file is parsed.
3. Dataclasses are built and validated.
4. Relative paths are resolved against the config file directory.

## Required Fields

These fields are required at the top level:

- `roots` (list of directories to scan)

Everything else has defaults.

## Top-Level Ingest Settings

These settings live at the top level of the config file and map to
`IngestConfig`.

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `roots` | list[path] | required | Root directories to scan. |
| `embed_model` | string | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for ingest/query. |
| `device` | string | `cpu` | Embedding device, e.g. `cpu`, `cuda`, `cuda:0`. |
| `batch_size` | int | `32` | Embedding batch size. |
| `persist_dir` | path | `.chroma` | ChromaDB storage directory. |
| `collection` | string | `local_kb` | ChromaDB collection name. |
| `chroma_flush_interval` | int | `2000` | Flush interval for Chroma writes. |
| `manifest` | path | `.ingest/manifest.json` | Incremental ingest manifest. |
| `extensions` | list[string] | default set | File extensions to include. |
| `override_extensions` | bool | `false` | Replace default extensions instead of merging. |
| `exclude_dirs` | list[string] | default set | Directory names to skip. |
| `override_excludes` | bool | `false` | Replace default excludes instead of merging. |
| `chunk_size` | int | `1800` | Characters per chunk. |
| `chunk_overlap` | int | `200` | Overlap between chunks. |
| `enable_ocr` | bool | `false` | OCR for image-based PDFs. |
| `enable_ai_tagging` | bool | `false` | Run LLM tagging during ingest. |
| `tagging_model` | string | `gpt-5.2` | LLM model for tagging. |
| `tags_per_chunk` | int | `3` | Tags to generate per chunk. |
| `dry_run` | bool | `false` | Simulate ingest without writing. |
| `skip_errors` | bool | `true` | Continue if a file/page fails to parse. |

Default extensions include: `.txt`, `.md`, `.rst`, `.pdf`, `.docx`, `.html`, `.htm`.

Default excluded directory names include: `.git`, `.hg`, `.svn`, `__pycache__`,
`.venv`, `venv`, `node_modules`.

## LLM Configuration

The `llm` section controls provider settings and optional per-feature overrides.

```json
"llm": {
  "provider": "openai",
  "model": "gpt-5.2",
  "api_key": null,
  "temperature": 0.7,
  "max_tokens": 2000,
  "query": {
    "model": "gpt-5.2",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "tagging": {
    "model": "gpt-4o-mini",
    "temperature": 0.5,
    "max_tokens": 200
  },
  "ranking": {
    "model": "gpt-5.2",
    "temperature": 0.3,
    "max_tokens": 500
  }
}
```

Notes:

- `api_key` can be omitted and loaded from environment variables.
- Per-feature overrides inherit from the base `llm` config.
- Supported provider today: `openai`.

### Legacy OpenAI Section

A legacy `openai` section is still supported for API key loading:

```json
"openai": { "api_key": "..." }
```

Prefer `llm.api_key` going forward.

## Query Configuration

The `query` section controls retrieval and reranking behavior.

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `mode` | string | `grounded` | Active mode name. |
| `k` | int | `12` | Initial retrieval size. |
| `retrieve_k` | int? | null | Override retrieval size when filtering. |
| `min_relevance` | float | `0.25` | Minimum relevance to synthesize. |
| `k_rerank` | int | `6` | Final candidate count after reranking. |
| `rerank_strategy` | string | `hybrid` | `none`, `lexical`, `llm`, `hybrid`. |
| `no_rerank` | bool | `false` | Disable LLM reranking even in `hybrid`. |
| `local_pool` | int? | null | Pool size before diversity enforcement. |
| `max_per_file` | int | `2` | Max chunks per source file. |
| `path_contains` | list[string]? | null | Filter: substring(s) in path. |
| `ext_allow` | list[string]? | null | Filter: allow extensions. |
| `ext_deny` | list[string]? | null | Filter: deny extensions. |

`local_pool` defaults to `max(k * 3, k_rerank * 4)` when not provided.

## Modes Configuration

Modes define per-mode behavior for sources and notes.

```json
"modes": {
  "research": {
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
}
```

If `modes` is not defined, LSM uses built-in `grounded`, `insight`, and `hybrid`.

## Remote Providers

Remote providers live in `remote_providers` and are used when a mode enables
remote sources.

```json
"remote_providers": {
  "brave": {
    "type": "web_search",
    "enabled": true,
    "weight": 1.0,
    "api_key": "...",
    "max_results": 5
  }
}
```

Provider `type` must be registered by the remote provider factory. Built-ins:

- `web_search`
- `brave_search`

## Environment Variables

Common environment variables:

- `OPENAI_API_KEY` (OpenAI provider)
- `BRAVE_API_KEY` (Brave Search provider)
- `<PROVIDER>_API_KEY` (generic convention for future providers)

If `llm.api_key` is omitted, LSM reads `OPENAI_API_KEY` or
`{PROVIDER}_API_KEY` automatically.

## Path Resolution Rules

- `persist_dir` and `manifest` are resolved relative to the config file.
- Notes directories are resolved relative to the config file when notes are
  written from query mode.
- `roots` can be absolute or relative to the working directory.

## Common Configuration Scenarios

### Minimal Config

```json
{
  "roots": ["C:/Users/You/Documents"],
  "llm": { "api_key": null }
}
```

### Query-Only (No LLM Rerank)

```json
{
  "roots": ["C:/Users/You/Documents"],
  "query": {
    "rerank_strategy": "lexical",
    "no_rerank": true
  },
  "llm": { "api_key": null }
}
```

### OCR-Enabled PDF Ingest

```json
{
  "roots": ["C:/Users/You/Scans"],
  "enable_ocr": true,
  "llm": { "api_key": null }
}
```

### Remote Sources Enabled (Hybrid Mode)

```json
{
  "roots": ["C:/Users/You/Documents"],
  "modes": {
    "hybrid": {
      "synthesis_style": "grounded",
      "source_policy": {
        "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
        "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 5 },
        "model_knowledge": { "enabled": true, "require_label": true }
      },
      "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
    }
  },
  "remote_providers": {
    "brave": {
      "type": "web_search",
      "enabled": true,
      "api_key": "${BRAVE_API_KEY}",
      "max_results": 5
    }
  },
  "llm": { "api_key": null }
}
```

## Troubleshooting Configuration

- If LSM says `query.mode` is not found, ensure it matches a key in `modes`.
- If Chroma errors occur, verify `persist_dir` is writable and not empty.
- If no files are found, double-check `roots`, `extensions`, and `exclude_dirs`.
- If OpenAI is not available, set `OPENAI_API_KEY` in `.env` or `llm.api_key`.
- For Brave Search, set `BRAVE_API_KEY` or `remote_providers.<name>.api_key`.
