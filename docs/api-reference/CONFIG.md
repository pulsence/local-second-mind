# Configuration API Reference

This document describes the configuration dataclasses in `lsm/config/models/`.

## LSMConfig

Top-level configuration container.

Fields:

- `ingest: IngestConfig` (required)
- `query: QueryConfig` (required)
- `llm: LLMRegistryConfig` (required)
- `modes: list[ModeConfig] | None` (optional, built-ins if None)
- `remote_providers: list[RemoteProviderConfig] | None`
- `config_path: Path | None` (used for path resolution)

Methods:

- `validate()`
- `get_mode_config(mode_name=None)`
- `get_active_remote_providers(allowed_names=None)`

## IngestConfig

Controls ingestion behavior.

Fields:

- `roots: list[Path]` (required)
- `embed_model: str = sentence-transformers/all-MiniLM-L6-v2`
- `device: str = cpu`
- `batch_size: int = 32`
- `persist_dir: Path = .chroma`
- `collection: str = local_kb`
- `chroma_flush_interval: int = 2000`
- `manifest: Path = .ingest/manifest.json`
- `extensions: list[str] | None`
- `override_extensions: bool = false`
- `exclude_dirs: list[str] | None`
- `override_excludes: bool = false`
- `chunk_size: int = 1800`
- `chunk_overlap: int = 200`
- `enable_ocr: bool = false`
- `enable_ai_tagging: bool = false`
- `tagging_model: str = gpt-5.2`
- `tags_per_chunk: int = 3`
- `dry_run: bool = false`
- `skip_errors: bool = true`

Derived properties:

- `exts: set[str]` (defaults + `extensions`)
- `exclude_set: set[str]` (defaults + `exclude_dirs`)

## QueryConfig

Controls retrieval and reranking.

Fields:

- `k: int = 12`
- `retrieve_k: int | None = None`
- `min_relevance: float = 0.25`
- `k_rerank: int = 6`
- `rerank_strategy: str = hybrid`
- `no_rerank: bool = false`
- `local_pool: int | None = None`
- `max_per_file: int = 2`
- `mode: str = grounded`
- `path_contains: list[str] | None = None`
- `ext_allow: list[str] | None = None`
- `ext_deny: list[str] | None = None`

Derived behavior:

- `local_pool` defaults to `max(k * 3, k_rerank * 4)`.
- `no_rerank` forces `rerank_strategy = none`.

## LLMRegistryConfig

Ordered list of LLM providers with per-feature selection.

Fields:

- `llms: list[LLMProviderConfig]`

Methods:

- `validate()`
- `get_query_config()`
- `get_tagging_config()`
- `get_ranking_config()`
- `get_feature_provider_map()`
- `get_provider_names()`
- `get_provider_by_name(name)`
- `override_feature_model(feature_name, model)`

## LLMProviderConfig

Provider entry inside the registry.

Fields:

- `provider_name: str`
- `api_key: str | None = None`
- `base_url: str | None = None` (local/hosted providers)
- `endpoint: str | None = None` (Azure OpenAI)
- `api_version: str | None = None` (Azure OpenAI)
- `deployment_name: str | None = None` (Azure OpenAI)
- `query: FeatureLLMConfig | None`
- `tagging: FeatureLLMConfig | None`
- `ranking: FeatureLLMConfig | None`

Notes:

- `model`, `temperature`, and `max_tokens` are feature-level only.
- Configure them under `query`, `tagging`, and/or `ranking`.

## LLMConfig

Effective LLM configuration used at runtime.

Fields:

- `provider: str`
- `model: str`
- `api_key: str | None = None`
- `temperature: float = 0.7`
- `max_tokens: int = 2000`
- `base_url: str | None = None` (local/hosted providers)
- `endpoint: str | None = None` (Azure OpenAI)
- `api_version: str | None = None` (Azure OpenAI)
- `deployment_name: str | None = None` (Azure OpenAI)

Methods:

- `validate()`

Environment resolution:

- If `api_key` is not set, LSM reads `{PROVIDER}_API_KEY`.
- For `gemini`, `GOOGLE_API_KEY` is supported.

## FeatureLLMConfig


Per-feature override used to build effective runtime `LLMConfig`.

Fields:

- `model: str | None`
- `api_key: str | None`
- `temperature: float | None`
- `max_tokens: int | None`
- `base_url: str | None`
- `endpoint: str | None`
- `api_version: str | None`
- `deployment_name: str | None`

Method:

- `merge_with_base(base: LLMConfig) -> LLMConfig`

## ModeConfig

Defines a single mode.

Fields:

- `synthesis_style: str = grounded` (`grounded` or `insight`)
- `source_policy: SourcePolicyConfig`
- `notes: NotesConfig`

Method:

- `validate()`

## SourcePolicyConfig

Groups source policies:

- `local: LocalSourcePolicy`
- `remote: RemoteSourcePolicy`
- `model_knowledge: ModelKnowledgePolicy`

## LocalSourcePolicy

Fields:

- `enabled: bool = true`
- `min_relevance: float = 0.25`
- `k: int = 12`
- `k_rerank: int = 6`

## RemoteSourcePolicy

Fields:

- `enabled: bool = false`
- `rank_strategy: str = weighted`
- `max_results: int = 5`
- `remote_providers: list[str | RemoteProviderRef] | None = None`

## ModelKnowledgePolicy

Fields:

- `enabled: bool = false`
- `require_label: bool = true`

## NotesConfig

Fields:

- `enabled: bool = true`
- `dir: str = notes`
- `template: str = default`
- `filename_format: str = timestamp`

## RemoteProviderConfig

Fields:

- `name: str` (required)
- `type: str` (required)
- `weight: float = 1.0`
- `api_key: str | None = None`
- `endpoint: str | None = None`
- `max_results: int | None = None`
- `language: str | None = None`
- `user_agent: str | None = None`
- `timeout: int | None = None`
- `min_interval_seconds: float | None = None`
- `section_limit: int | None = None`
- `snippet_max_chars: int | None = None`
- `include_disambiguation: bool | None = None`
