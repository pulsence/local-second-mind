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
| `llms` | `LLMRegistryConfig` | LLM providers, tiers, and services |
| `query` | `QueryConfig` | Retrieval and reranking settings |
| `modes` | `list[ModeConfig]` | Query mode definitions |
| `notes` | `NotesConfig` | Notes system settings |
| `chats` | `ChatsConfig` | Chat transcript saving settings |
| `remote_providers` | `list[RemoteProviderConfig]` | Remote source providers |
| `remote_provider_chains` | `list[RemoteProviderChainConfig]` | Named multi-provider remote pipelines |
| `agents` | `AgentConfig` | Agent runtime, sandbox, memory, scheduling, and interaction settings |

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
| `enable_ai_tagging` | bool | `false` | Run LLM tagging during ingest. Tagging model is configured via `llms.services.tagging`. |
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

The `llms` section is an object with `providers` (connection details),
optional `tiers` (quick/normal/complex defaults), and `services`
(feature-to-model mappings).

```json
"llms": {
  "providers": [
    { "provider_name": "openai", "api_key": null },
    { "provider_name": "gemini", "api_key": null },
    { "provider_name": "claude", "api_key": null }
  ],
  "tiers": {
    "quick": { "provider": "gemini", "model": "gemini-2.5-flash-lite" },
    "normal": { "provider": "openai", "model": "gpt-5.2" },
    "complex": { "provider": "claude", "model": "claude-haiku-4-5" }
  },
  "services": {
    "default": { "provider": "openai", "model": "gpt-5.2" },
    "query":   { "provider": "openai", "model": "gpt-5.2", "temperature": 0.7, "max_tokens": 2000 },
    "decomposition": { "provider": "openai", "model": "gpt-5-nano", "temperature": 0.2, "max_tokens": 300 },
    "tagging": { "provider": "gemini", "model": "gemini-2.5-flash-lite", "temperature": 0.5, "max_tokens": 200 },
    "ranking": { "provider": "claude", "model": "claude-haiku-4-5", "temperature": 0.3, "max_tokens": 500 }
  }
}
```

Notes:

- `providers[]` entries hold connection/auth settings only:
  `provider_name`, `api_key`, `base_url`, `endpoint`, `api_version`, `deployment_name`.
- `tiers` defines reusable provider+model defaults for capability tiers used by agents/tools.
- `services` maps logical names to a provider + model combination with optional
  `temperature` and `max_tokens` overrides.
- A `"query"` or `"default"` service is required. `tagging`, `ranking`, and
  `decomposition` are
  optional and fall back to `"default"` if not defined.
- `decomposition` controls which model is used for natural-language query decomposition.
- Each service's `provider` field (when set) must reference a name from `providers[]`.
- Supported providers: `openai`, `openrouter`, `anthropic`, `gemini`, `local`, `azure_openai`.

Provider-specific fields:

- `base_url` is used for local/hosted providers (e.g., Ollama).
- `endpoint`, `api_version`, and `deployment_name` are used for Azure OpenAI.
- `fallback_models` (optional) configures OpenRouter model routing fallbacks.

### Provider Examples

OpenRouter:

```json
"llms": {
  "providers": [
    {
      "provider_name": "openrouter",
      "fallback_models": ["anthropic/claude-3.5-sonnet"]
    }
  ],
  "services": {
    "query": { "provider": "openrouter", "model": "openai/gpt-4o" }
  }
}
```

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
| `enable_query_cache` | bool | `false` | Enable local in-memory query result cache (TTL + LRU). |
| `query_cache_ttl` | int | `3600` | Local query cache TTL in seconds. |
| `query_cache_size` | int | `100` | Max entries in local query cache. |
| `chat_mode` | string | `single` | Response mode: `single` or `chat`. |
| `enable_llm_server_cache` | bool | `true` | Enable provider-side prompt/session cache reuse for chat follow-up turns. |

`local_pool` defaults to `max(k * 3, k_rerank * 4)` when not provided.

Query caching notes:

- `enable_query_cache` caches synthesized query results in process memory with query/mode/filter-sensitive cache keys.
- `enable_llm_server_cache` enables provider server-side cache/session reuse in `chat_mode = "chat"`.
- LSM tracks provider response/session IDs and reuses them on follow-up turns when the provider API supports this.
- Provider cache retention is managed by provider backends; LSM does not expose a retention duration setting.

Context-building notes:

- Query execution applies metadata prefiltering before vector similarity search when matching metadata exists.
- Prefiltering considers `content_type`, `ai_tags`, `user_tags`, `root_tags`, `folder_tags`, plus deterministic author/year/title hints extracted from the prompt.
- Context anchors (`/context doc ...` and `/context chunk ...` in TUI) are session-level controls, not static config keys.

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
    },
    "chats": {
      "auto_save": true,
      "dir": "Chats/Research"
    }
  }
]
```

If `modes` is not defined, LSM uses built-in `grounded`, `insight`, and `hybrid`.

Mode chat overrides:

- `modes[].chats.auto_save` overrides global `chats.auto_save` for that mode.
- `modes[].chats.dir` overrides global `chats.dir` for that mode.
- Chat transcripts are saved under the mode subfolder (for example, `Chats/Research/grounded`).

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

## Global Chats Configuration

Chats are configured once at the top level and can be overridden by modes.

```json
"chats": {
  "enabled": true,
  "dir": "Chats",
  "auto_save": true,
  "format": "markdown"
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
    "max_results": 5,
    "cache_results": false,
    "cache_ttl": 86400
  }
]
```

- `cache_results`: When `true`, provider responses are cached on disk under
  `<global_folder>/Downloads/<provider_name>/`.
- `cache_ttl`: Cache freshness window in seconds before a query is re-fetched.

## Remote Provider Chains

Remote provider chains define multi-step pipelines where each step can map
fields from the previous step's output into the next step's input.

```json
"remote_provider_chains": [
  {
    "name": "Research Digest",
    "agent_description": "Use OpenAlex to discover works, then enrich DOI metadata via Crossref.",
    "links": [
      { "source": "openalex" },
      { "source": "crossref", "map": ["doi:doi"] }
    ]
  }
]
```

- `links[0]` receives the chain input directly.
- Later links can declare `map` entries as `"output_field:input_field"`.

Provider `type` must be registered by the remote provider factory. Built-ins:

- `web_search`
- `brave_search`
- `wikipedia`
- `arxiv`

## Agents Configuration

Agents are configured under the top-level `agents` key.

```json
"agents": {
  "enabled": true,
  "agents_folder": "Agents",
  "max_tokens_budget": 200000,
  "max_iterations": 25,
  "max_concurrent": 5,
  "log_stream_queue_limit": 500,
  "context_window_strategy": "compact",
  "sandbox": {
    "allowed_read_paths": ["./docs", "./notes", "./.ingest", "./.chroma"],
    "allowed_write_paths": ["./Agents", "./notes"],
    "allow_url_access": true,
    "require_user_permission": {},
    "require_permission_by_risk": {},
    "execution_mode": "local_only",
    "force_docker": false,
    "command_allowlist": ["ls", "cat"],
    "command_denylist": ["rm"],
    "wsl2": { "enabled": false, "distro": "Ubuntu" }
  },
  "memory": {
    "enabled": true,
    "storage_backend": "auto",
    "sqlite_path": "memory.sqlite3"
  },
  "interaction": {
    "timeout_seconds": 300,
    "timeout_action": "deny",
    "auto_continue": false
  }
}
```

| Key | Type | Default | Purpose |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the agent runtime. |
| `agents_folder` | path | `Agents` | Folder for agent runs, logs, and scheduler state. |
| `max_tokens_budget` | int | `200000` | Per-run token budget cap. |
| `max_iterations` | int | `25` | Per-run action-loop cap. |
| `max_concurrent` | int | `5` | Maximum simultaneous agent runs. |
| `log_stream_queue_limit` | int | `500` | Max buffered live-log entries per running agent before oldest entries are dropped. |
| `context_window_strategy` | string | `compact` | Agent loop context strategy: `compact` or `fresh`. |
| `sandbox` | object | see defaults | Tool path/network/permission/runner restrictions. |
| `memory` | object | see defaults | Memory backend + TTL configuration. |
| `interaction` | object | see defaults | Interaction request timeout policy. |
| `agent_configs` | object | `{}` | Per-agent override dictionary keyed by agent name. |
| `schedules` | list | `[]` | Optional scheduler entries. |

`agents.interaction` fields:

- `timeout_seconds` (`int`, default `300`): max seconds to wait for user interaction response.
- `timeout_action` (`string`, default `deny`): timeout fallback; `deny` raises a permission error, `approve` auto-approves.
- `auto_continue` (`bool`, default `false`): auto-respond to `ask_user` prompts with "Continue with your best judgment."

Per-agent LLM overrides (under `agents.agent_configs.<agent_name>`):

- `llm_tier`: use a named tier (e.g., `quick`, `normal`, `complex`)
- `llm_service`: use a named service from `llms.services`
- `llm_provider` + `llm_model`: override provider/model directly
- `llm_temperature`, `llm_max_tokens`: optional runtime overrides

`agents.sandbox` highlights:

- `execution_mode`: `local_only` or `prefer_docker` (prefers Docker/WSL2 for exec+network tools).
- `force_docker`: require Docker for all tool risks; blocks execution when Docker is unavailable.
- `limits`: `timeout_s_default`, `max_stdout_kb`, `max_file_write_mb`.
- `docker`: `enabled`, `image`, `network_default`, `cpu_limit`, `mem_limit_mb`, `read_only_root`.
- `wsl2`: `enabled`, `distro`, `wsl_bin`, `shell`.
- `command_allowlist`: optional list of allowed command prefixes for `bash`/`powershell`.
- `command_denylist`: optional list of blocked command prefixes for `bash`/`powershell`.

## Environment Variables

Common environment variables:

- `OPENAI_API_KEY` (OpenAI provider)
- `OPENROUTER_API_KEY` (OpenRouter provider)
- `ANTHROPIC_API_KEY` (Anthropic provider)
- `GOOGLE_API_KEY` (Gemini provider)
- `AZURE_OPENAI_API_KEY` (Azure OpenAI provider)
- `AZURE_OPENAI_ENDPOINT` (Azure OpenAI endpoint)
- `AZURE_OPENAI_API_VERSION` (Azure OpenAI API version)
- `AZURE_OPENAI_DEPLOYMENT_NAME` (Azure OpenAI deployment name)
- `OPENROUTER_APP_NAME` (optional OpenRouter app attribution)
- `OPENROUTER_APP_URL` (optional OpenRouter app attribution)
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
- `agents.agents_folder` is resolved relative to `global.global_folder` when
  provided as a relative path.
- `agents.memory.sqlite_path` is resolved relative to `agents.agents_folder`
  when provided as a relative path.
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
