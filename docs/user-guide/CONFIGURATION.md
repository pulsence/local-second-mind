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

## Config Structure

All settings are nested under section objects. There are zero flat top-level
fields in the config file. The top-level keys are:

| Key | Dataclass | Purpose |
| --- | --- | --- |
| `global` | `GlobalConfig` | Shared settings (embedding model, device, batch size, global folder) |
| `ingest` | `IngestConfig` | Ingestion pipeline settings |
| `vectordb` | `VectorDBConfig` | Vector database provider settings |
| `llms` | `LLMRegistryConfig` | LLM providers and services |
| `query` | `QueryConfig` | Retrieval and reranking settings |
| `modes` | `list[ModeConfig]` | Query mode definitions |
| `notes` | `NotesConfig` | Notes system settings |
| `remote_providers` | `list[RemoteProviderConfig]` | Remote source providers |

## Required Fields

The only required field is `ingest.roots` (list of directories to scan).
Everything else has defaults.

## Global Settings

Global settings live under the `"global"` key and map to `GlobalConfig`
(`lsm/config/models/global_config.py`). These are shared across multiple modules
(ingest, query, agents, etc.).

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `global_folder` | path | `<HOME>/Local Second Mind` | Global folder for notes, chats, agents. |
| `embed_model` | string | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for ingest/query. |
| `device` | string | `cpu` | Embedding device, e.g. `cpu`, `cuda`, `cuda:0`, `mps`. |
| `batch_size` | int | `32` | Embedding batch size. |

Environment variable overrides: `LSM_GLOBAL_FOLDER`, `LSM_EMBED_MODEL`,
`LSM_DEVICE`.

## Ingest Settings

Ingest settings live under the `"ingest"` key and map to `IngestConfig`.

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `roots` | list[path] | required | Root directories to scan. |
| `persist_dir` | path | `.chroma` | ChromaDB persistent storage directory. |
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

## Vector DB Configuration

Vector DB settings live under `vectordb`.

```json
"vectordb": {
  "provider": "chromadb",
  "persist_dir": ".chroma",
  "collection": "local_kb",
  "chroma_hnsw_space": "cosine"
}
```

## LLM Configuration

The `llms` section is an object with two keys: `providers` (connection details)
and `services` (feature-to-model mappings).

```json
"llms": {
  "providers": [
    { "provider_name": "openai", "api_key": null },
    { "provider_name": "gemini", "api_key": null },
    { "provider_name": "claude", "api_key": null }
  ],
  "services": {
    "default": { "provider": "openai", "model": "gpt-5.2" },
    "query":   { "provider": "openai", "model": "gpt-5.2", "temperature": 0.7, "max_tokens": 2000 },
    "tagging": { "provider": "gemini", "model": "gemini-2.5-flash-lite", "temperature": 0.5, "max_tokens": 200 },
    "ranking": { "provider": "claude", "model": "claude-haiku-4-5", "temperature": 0.3, "max_tokens": 500 }
  }
}
```

Notes:

- `providers[]` entries hold connection/auth settings only:
  `provider_name`, `api_key`, `base_url`, `endpoint`, `api_version`, `deployment_name`.
- `services` maps logical names to a provider + model combination with optional
  `temperature` and `max_tokens` overrides.
- A `"query"` or `"default"` service is required. `tagging` and `ranking` are
  optional and fall back to `"default"` if not defined.
- Each service's `provider` field must reference a name from `providers[]`.
- Supported providers: `openai`, `anthropic`, `gemini`, `local`, `azure_openai`.

Provider-specific fields:

- `base_url` is used for local/hosted providers (e.g., Ollama).
- `endpoint`, `api_version`, and `deployment_name` are used for Azure OpenAI.

### Provider Examples

Anthropic:

```json
"llms": {
  "providers": [{ "provider_name": "anthropic" }],
  "services": {
    "query":   { "provider": "anthropic", "model": "claude-3-5-sonnet-20241022" },
    "tagging": { "provider": "anthropic", "model": "claude-3-5-haiku-20241022" },
    "ranking": { "provider": "anthropic", "model": "claude-3-5-haiku-20241022" }
  }
}
```

Local (Ollama):

```json
"llms": {
  "providers": [{ "provider_name": "local", "base_url": "http://localhost:11434" }],
  "services": {
    "default": { "provider": "local", "model": "llama3" }
  }
}
```

Gemini:

```json
"llms": {
  "providers": [{ "provider_name": "gemini" }],
  "services": {
    "query":   { "provider": "gemini", "model": "gemini-1.5-pro" },
    "tagging": { "provider": "gemini", "model": "gemini-1.5-flash" },
    "ranking": { "provider": "gemini", "model": "gemini-1.5-pro" }
  }
}
```

Azure OpenAI:

```json
"llms": {
  "providers": [{
    "provider_name": "azure_openai",
    "endpoint": "https://your-resource.openai.azure.com/",
    "api_version": "2023-05-15",
    "deployment_name": "gpt-35-turbo"
  }],
  "services": {
    "default": { "provider": "azure_openai", "model": "gpt-35-turbo" }
  }
}
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

Modes define per-mode behavior for synthesis and source blending.

```json
"modes": [
  {
    "name": "research",
    "synthesis_style": "grounded",
    "source_policy": {
      "local": { "enabled": true, "min_relevance": 0.20, "k": 15, "k_rerank": 8 },
      "remote": {
        "enabled": true,
        "rank_strategy": "weighted",
        "max_results": 10,
        "remote_providers": ["brave", "wikipedia", "arxiv"]
      },
      "model_knowledge": { "enabled": true, "require_label": true }
    }
  }
]
```

If `modes` is not defined, LSM uses built-in `grounded`, `insight`, and `hybrid`.

## Global Notes Configuration

Notes are configured once at the top level and apply to all modes.

```json
"notes": {
  "enabled": true,
  "dir": "notes",
  "template": "default",
  "filename_format": "timestamp",
  "integration": "none",
  "wikilinks": false,
  "backlinks": false,
  "include_tags": false
}
```

## Remote Providers

Remote providers live in `remote_providers` and are used when a mode enables
remote sources. Use `source_policy.remote.remote_providers` to restrict a mode
to specific providers.

```json
"remote_providers": [
  {
    "name": "brave",
    "type": "web_search",
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
- `arxiv`

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
- `LSM_ARXIV_USER_AGENT` (arXiv provider User-Agent)
- `<PROVIDER>_API_KEY` (generic convention for future providers)

If a provider entry omits `api_key`, LSM reads `{PROVIDER}_API_KEY`
automatically (and `GOOGLE_API_KEY` for Gemini). Local providers do not require
an API key.

## Path Resolution Rules

- `vectordb.persist_dir`, `ingest.persist_dir`, and `ingest.manifest` are
  resolved relative to the config file.
- If `notes.dir` is `"notes"`, notes are written to `<global_folder>/Notes`.
- For any other relative `notes.dir`, paths are resolved relative to the config file.
- `ingest.roots` can be absolute or relative to the working directory.

## Common Configuration Scenarios

### Minimal Config

```json
{
  "ingest": {
    "roots": ["C:/Users/You/Documents"]
  },
  "llms": {
    "providers": [{ "provider_name": "openai" }],
    "services": {
      "query": { "provider": "openai", "model": "gpt-5.2" }
    }
  }
}
```

### Query-Only (No LLM Rerank)

```json
{
  "ingest": {
    "roots": ["C:/Users/You/Documents"]
  },
  "query": {
    "rerank_strategy": "lexical",
    "no_rerank": true
  },
  "llms": {
    "providers": [{ "provider_name": "openai" }],
    "services": {
      "query": { "provider": "openai", "model": "gpt-5.2" }
    }
  }
}
```

### OCR-Enabled PDF Ingest

```json
{
  "ingest": {
    "roots": ["C:/Users/You/Scans"],
    "enable_ocr": true
  },
  "llms": {
    "providers": [{ "provider_name": "openai" }],
    "services": {
      "query": { "provider": "openai", "model": "gpt-5.2" }
    }
  }
}
```

### Remote Sources Enabled (Hybrid Mode)

```json
{
  "ingest": {
    "roots": ["C:/Users/You/Documents"]
  },
  "vectordb": {
    "provider": "chromadb",
    "persist_dir": ".chroma",
    "collection": "local_kb"
  },
  "modes": [
    {
      "name": "hybrid",
      "synthesis_style": "grounded",
      "source_policy": {
        "local": { "min_relevance": 0.25, "k": 12, "k_rerank": 6 },
        "remote": {
          "enabled": true,
          "rank_strategy": "weighted",
          "max_results": 5,
          "remote_providers": ["wikipedia", "arxiv"]
        },
        "model_knowledge": { "enabled": true, "require_label": true }
    }
  ],
  "notes": {
    "enabled": true,
    "dir": "notes",
    "template": "default",
    "filename_format": "timestamp"
  },
  "remote_providers": [
    {
      "name": "brave",
      "type": "web_search",
      "api_key": "${BRAVE_API_KEY}",
      "max_results": 5
    },
    {
      "name": "wikipedia",
      "type": "wikipedia",
      "language": "en",
      "user_agent": "${LSM_WIKIPEDIA_USER_AGENT}",
      "max_results": 5
    },
    {
      "name": "arxiv",
      "type": "arxiv",
      "user_agent": "${LSM_ARXIV_USER_AGENT}",
      "max_results": 5
    }
  ],
  "llms": {
    "providers": [{ "provider_name": "openai" }],
    "services": {
      "query": { "provider": "openai", "model": "gpt-5.2" }
    }
  }
}
```

## Troubleshooting Configuration

- If LSM says `query.mode` is not found, ensure it matches a `modes` entry name.
- If Chroma errors occur, verify `persist_dir` is writable and not empty.
- If no files are found, double-check `ingest.roots`, `ingest.extensions`, and `ingest.exclude_dirs`.
- If OpenAI is not available, set `OPENAI_API_KEY` in `.env` or in the provider's `api_key` field.
- For Brave Search, set `BRAVE_API_KEY` or the `remote_providers` entry `api_key`.
- For Wikipedia, set `LSM_WIKIPEDIA_USER_AGENT` or the `remote_providers` entry `user_agent`.
