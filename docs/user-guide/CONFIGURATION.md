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

The `llms` section defines an ordered list of provider entries. Later entries
override feature selection when they define a feature (`query`, `tagging`,
`ranking`).

```json
"llms": [
  {
    "provider_name": "openai",
    "api_key": null,
    "query": {
      "model": "gpt-5.2",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "tagging": {
      "model": "gpt-5-nano",
      "temperature": 0.5,
      "max_tokens": 200
    },
    "ranking": {
      "model": "gpt-5-nano",
      "temperature": 0.3,
      "max_tokens": 500
    }
  },
  {
    "provider_name": "gemini",
    "api_key": null,
    "tagging": {
      "model": "gemini-2.5-flash-lite",
      "temperature": 1,
      "max_tokens": 200
    }
  },
  {
    "provider_name": "claude",
    "api_key": null,
    "ranking": {
      "model": "claude-haiku-4-5",
      "temperature": 1,
      "max_tokens": 200
    }
  }
]
```

Notes:

- Each `llms[]` entry must define at least one feature: `query`, `tagging`, or
  `ranking`.
- For each feature, the last provider in the list that defines it is selected.
- `query` is required. `tagging` and `ranking` are optional and only needed if
  you use those features.
- Provider-level fields (`api_key`, `model`, `temperature`, `max_tokens`,
  `base_url`, `endpoint`, `api_version`, `deployment_name`) apply to feature
  configs unless overridden.
- Supported providers: `openai`, `anthropic`, `gemini`, `local`, `azure_openai`.

Provider-specific fields:

- `base_url` is used for local/hosted providers (e.g., Ollama).
- `endpoint`, `api_version`, and `deployment_name` are used for Azure OpenAI.

### Provider Examples

Anthropic:

```json
"llms": [
  {
    "provider_name": "anthropic",
    "query": { "model": "claude-3-5-sonnet-20241022" },
    "tagging": { "model": "claude-3-5-haiku-20241022" },
    "ranking": { "model": "claude-3-5-haiku-20241022" }
  }
]
```

Local (Ollama):

```json
"llms": [
  {
    "provider_name": "local",
    "base_url": "http://localhost:11434",
    "query": { "model": "llama3" },
    "tagging": { "model": "llama3" },
    "ranking": { "model": "llama3" }
  }
]
```

Gemini:

```json
"llms": [
  {
    "provider_name": "gemini",
    "query": { "model": "gemini-1.5-pro" },
    "tagging": { "model": "gemini-1.5-flash" },
    "ranking": { "model": "gemini-1.5-pro" }
  }
]
```

Azure OpenAI:

```json
"llms": [
  {
    "provider_name": "azure_openai",
    "api_key": "${AZURE_OPENAI_API_KEY}",
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_version": "2023-05-15",
    "deployment_name": "gpt-35-turbo",
    "query": { "model": "gpt-35-turbo" },
    "tagging": { "model": "gpt-35-turbo" },
    "ranking": { "model": "gpt-35-turbo" }
  }
]
```

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

If `modes` is not defined, LSM uses built-in `grounded`, `insight`, and `hybrid`.

## Remote Providers

Remote providers live in `remote_providers` and are used when a mode enables
remote sources.

```json
"remote_providers": [
  {
    "name": "brave",
    "type": "web_search",
    "enabled": true,
    "weight": 1.0,
    "api_key": "...",
    "max_results": 5
  }
]
```

Provider `type` must be registered by the remote provider factory. Built-ins:

- `web_search`
- `brave_search`
- `wikipedia`

## Environment Variables

Common environment variables:

- `OPENAI_API_KEY` (OpenAI provider)
- `ANTHROPIC_API_KEY` (Anthropic provider)
- `GOOGLE_API_KEY` (Gemini provider)
- `AZURE_OPENAI_API_KEY` (Azure OpenAI provider)
- `AZURE_OPENAI_ENDPOINT` (Azure OpenAI endpoint)
- `AZURE_OPENAI_API_VERSION` (Azure OpenAI API version)
- `AZURE_OPENAI_DEPLOYMENT_NAME` (Azure OpenAI deployment name)
- `OLLAMA_BASE_URL` (Local/Ollama base URL)
- `BRAVE_API_KEY` (Brave Search provider)
- `LSM_WIKIPEDIA_USER_AGENT` (Wikipedia provider User-Agent)
- `<PROVIDER>_API_KEY` (generic convention for future providers)

If a provider entry omits `api_key`, LSM reads `{PROVIDER}_API_KEY`
automatically (and `GOOGLE_API_KEY` for Gemini). Local providers do not require
an API key.

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
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" }
    }
  ]
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
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" }
    }
  ]
}
```

### OCR-Enabled PDF Ingest

```json
{
  "roots": ["C:/Users/You/Scans"],
  "enable_ocr": true,
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" }
    }
  ]
}
```

### Remote Sources Enabled (Hybrid Mode)

```json
{
  "roots": ["C:/Users/You/Documents"],
  "modes": [
    {
      "name": "hybrid",
      "synthesis_style": "grounded",
      "source_policy": {
        "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
        "remote": { "enabled": true, "rank_strategy": "weighted", "max_results": 5 },
        "model_knowledge": { "enabled": true, "require_label": true }
      },
      "notes": { "enabled": true, "dir": "notes", "template": "default", "filename_format": "timestamp" }
    }
  ],
  "remote_providers": [
    {
      "name": "brave",
      "type": "web_search",
      "enabled": true,
      "api_key": "${BRAVE_API_KEY}",
      "max_results": 5
    },
    {
      "name": "wikipedia",
      "type": "wikipedia",
      "enabled": true,
      "language": "en",
      "user_agent": "${LSM_WIKIPEDIA_USER_AGENT}",
      "max_results": 5
    }
  ],
  "llms": [
    {
      "provider_name": "openai",
      "query": { "model": "gpt-5.2" }
    }
  ]
}
```

## Troubleshooting Configuration

- If LSM says `query.mode` is not found, ensure it matches a `modes` entry name.
- If Chroma errors occur, verify `persist_dir` is writable and not empty.
- If no files are found, double-check `roots`, `extensions`, and `exclude_dirs`.
- If OpenAI is not available, set `OPENAI_API_KEY` in `.env` or `llms[].api_key`.
- For Brave Search, set `BRAVE_API_KEY` or the `remote_providers` entry `api_key`.
- For Wikipedia, set `LSM_WIKIPEDIA_USER_AGENT` or the `remote_providers` entry `user_agent`.
