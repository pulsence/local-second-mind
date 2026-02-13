"""
Configuration loader for Local Second Mind.

Handles loading configuration from JSON/YAML files and converting
to typed dataclass models.
"""

from __future__ import annotations

import json
import warnings
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from .models import (
    LSMConfig,
    AgentConfig,
    ScheduleConfig,
    MemoryConfig,
    SandboxConfig,
    GlobalConfig,
    IngestConfig,
    RootConfig,
    QueryConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    LLMServiceConfig,
    ModeConfig,
    ModeChatsConfig,
    SourcePolicyConfig,
    LocalSourcePolicy,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    NotesConfig,
    ChatsConfig,
    RemoteProviderConfig,
    RemoteProviderChainConfig,
    ChainLink,
    RemoteProviderRef,
    VectorDBConfig,
    DEFAULT_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
)

_REMOTE_PROVIDER_BASE_KEYS = {
    "name",
    "type",
    "weight",
    "api_key",
    "endpoint",
    "max_results",
    "language",
    "user_agent",
    "timeout",
    "min_interval_seconds",
    "section_limit",
    "snippet_max_chars",
    "include_disambiguation",
    "cache_results",
    "cache_ttl",
}


def load_raw_config(path: Path) -> Dict[str, Any]:
    """
    Load raw configuration from JSON or YAML file.

    Also loads environment variables from .env file if present.

    Args:
        path: Path to config file (.json, .yaml, or .yml)

    Returns:
        Dictionary with raw configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
        json.JSONDecodeError: If JSON is invalid
        yaml.YAMLError: If YAML is invalid
    """
    # Load environment variables
    load_dotenv()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    content = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        config = yaml.safe_load(content) or {}
    elif suffix == ".json":
        config = json.loads(content)
    else:
        raise ValueError(
            f"Unsupported config format: {suffix}. "
            f"Use .json, .yaml, or .yml"
        )

    return config


def build_llm_service_config(raw: Dict[str, Any]) -> LLMServiceConfig:
    """
    Build LLMServiceConfig from raw configuration.

    Args:
        raw: Raw service dictionary

    Returns:
        LLMServiceConfig instance
    """
    provider = raw.get("provider")
    model = raw.get("model")
    if not provider:
        raise ValueError("Each service must include 'provider'")
    if not model:
        raise ValueError("Each service must include 'model'")
    return LLMServiceConfig(
        provider=provider,
        model=model,
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
    )


def build_llm_provider_config(raw: Dict[str, Any]) -> LLMProviderConfig:
    """
    Build LLMProviderConfig from raw configuration.

    Args:
        raw: Raw provider dictionary

    Returns:
        LLMProviderConfig instance
    """
    provider_name = raw.get("provider_name")
    if not provider_name:
        raise ValueError("Each providers[] entry must include 'provider_name'")

    legacy_fields = {"model", "query", "tagging", "ranking"}
    found = legacy_fields & set(raw.keys())
    if found:
        raise ValueError(
            f"Legacy llms[] fields {sorted(found)} found on provider '{provider_name}'. "
            f"Move model/feature configs to the 'services' section instead."
        )

    return LLMProviderConfig(
        provider_name=provider_name,
        api_key=raw.get("api_key"),
        base_url=raw.get("base_url"),
        endpoint=raw.get("endpoint"),
        api_version=raw.get("api_version"),
        deployment_name=raw.get("deployment_name"),
    )


def build_llm_config(raw: Dict[str, Any]) -> LLMRegistryConfig:
    """
    Build LLMRegistryConfig from raw configuration.

    Args:
        raw: Raw config dictionary

    Returns:
        LLMRegistryConfig instance
    """
    llms_raw = raw.get("llms")
    if not llms_raw:
        raise ValueError("Config must include 'llms'")

    if isinstance(llms_raw, dict):
        providers_raw = llms_raw.get("providers")
        if not providers_raw or not isinstance(providers_raw, list):
            raise ValueError("llms.providers must be a non-empty list")

        services_raw = llms_raw.get("services")
        if not services_raw or not isinstance(services_raw, dict):
            raise ValueError("llms.services must be a non-empty dict")

        providers = [build_llm_provider_config(item) for item in providers_raw]
        services = {
            name: build_llm_service_config(svc)
            for name, svc in services_raw.items()
        }
        return LLMRegistryConfig(providers=providers, services=services)

    if isinstance(llms_raw, list):
        raise ValueError(
            "The 'llms' config format has changed. "
            "Use { 'providers': [...], 'services': {...} } instead of a list. "
            "See example_config.json for the new format."
        )

    raise ValueError("Config 'llms' must be an object with 'providers' and 'services' keys")


def build_global_config(raw: Dict[str, Any]) -> GlobalConfig:
    """
    Build GlobalConfig from raw configuration.

    Reads from the ``"global"`` sub-dict. Fields shared across multiple
    modules (embed_model, device, batch_size, global_folder) live here.

    Args:
        raw: Raw config dictionary

    Returns:
        GlobalConfig instance
    """
    global_raw = raw.get("global", {})

    global_folder = global_raw.get("global_folder")
    if global_folder is not None:
        global_folder = Path(global_folder)

    embedding_dimension_raw = global_raw.get("embedding_dimension")
    embedding_dimension = (
        int(embedding_dimension_raw)
        if embedding_dimension_raw is not None
        else None
    )

    return GlobalConfig(
        global_folder=global_folder,
        embed_model=global_raw.get("embed_model", GlobalConfig.embed_model),
        device=global_raw.get("device", GlobalConfig.device),
        batch_size=int(global_raw.get("batch_size", GlobalConfig.batch_size)),
        embedding_dimension=embedding_dimension,
    )


def build_ingest_config(raw: Dict[str, Any], config_path: Path) -> IngestConfig:
    """
    Build IngestConfig from raw configuration.

    Reads ingest-only fields from the ``"ingest"`` sub-dict.

    Args:
        raw: Raw config dictionary
        config_path: Path to config file (for resolving relative paths)

    Returns:
        IngestConfig instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    ingest_raw = raw.get("ingest", {})

    # Required field
    roots = ingest_raw.get("roots")
    if not roots or not isinstance(roots, list):
        raise ValueError("Config ingest section must include 'roots' as a non-empty list of directory paths")

    persist_dir = ingest_raw.get("persist_dir", IngestConfig.persist_dir)
    collection = ingest_raw.get("collection", IngestConfig.collection)

    # Build root configs (supports both strings and dicts)
    root_configs = []
    for r in roots:
        if isinstance(r, dict):
            root_configs.append(
                RootConfig(
                    path=Path(r["path"]),
                    tags=r.get("tags"),
                    content_type=r.get("content_type"),
                )
            )
        else:
            root_configs.append(RootConfig(path=Path(r)))

    # Build config
    config = IngestConfig(
        roots=root_configs,
        persist_dir=Path(persist_dir),
        collection=collection,
        chroma_flush_interval=int(ingest_raw.get("chroma_flush_interval", IngestConfig.chroma_flush_interval)),
        manifest=Path(ingest_raw.get("manifest", ".ingest/manifest.json")),
        extensions=ingest_raw.get("extensions"),
        override_extensions=bool(ingest_raw.get("override_extensions", False)),
        exclude_dirs=ingest_raw.get("exclude_dirs"),
        override_excludes=bool(ingest_raw.get("override_excludes", False)),
        chunk_size=int(ingest_raw.get("chunk_size", IngestConfig.chunk_size)),
        chunk_overlap=int(ingest_raw.get("chunk_overlap", IngestConfig.chunk_overlap)),
        chunking_strategy=str(ingest_raw.get("chunking_strategy", IngestConfig.chunking_strategy)),
        enable_ocr=bool(ingest_raw.get("enable_ocr", False)),
        enable_ai_tagging=bool(ingest_raw.get("enable_ai_tagging", False)),
        tags_per_chunk=int(ingest_raw.get("tags_per_chunk", 3)),
        dry_run=bool(ingest_raw.get("dry_run", False)),
        skip_errors=bool(ingest_raw.get("skip_errors", True)),
        enable_language_detection=bool(ingest_raw.get("enable_language_detection", False)),
        enable_translation=bool(ingest_raw.get("enable_translation", False)),
        translation_target=str(ingest_raw.get("translation_target", "en")),
        max_files=int(ingest_raw["max_files"]) if ingest_raw.get("max_files") is not None else None,
        max_seconds=int(ingest_raw["max_seconds"]) if ingest_raw.get("max_seconds") is not None else None,
        enable_versioning=bool(ingest_raw.get("enable_versioning", False)),
    )

    return config


def parse_config_text(content: str, path: Path | str) -> Dict[str, Any]:
    """
    Parse raw configuration content from JSON or YAML.

    Args:
        content: Config file content
        path: Path or filename used for extension detection
    """
    if isinstance(path, str):
        path = Path(path)

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(content) or {}
    if suffix == ".json":
        return json.loads(content)
    raise ValueError(
        f"Unsupported config format: {suffix}. "
        f"Use .json, .yaml, or .yml"
    )


def build_config_from_raw(raw: Dict[str, Any], path: Path | str) -> LSMConfig:
    """
    Build and validate LSMConfig from raw configuration data.
    """
    if isinstance(path, str):
        path = Path(path)

    path = path.expanduser().resolve()
    global_config = build_global_config(raw)
    llm_config = build_llm_config(raw)
    ingest_config = build_ingest_config(raw, path)
    query_config = build_query_config(raw)
    vectordb_config = build_vectordb_config(raw)
    modes = build_modes_registry(raw)
    notes_config = build_notes_config(raw.get("notes", {}))
    chats_config = build_chats_config(raw.get("chats", {}))
    agents_config = build_agent_config(raw.get("agents"))
    remote_providers = build_remote_providers_registry(raw)
    remote_provider_chains = build_remote_provider_chains_registry(raw)

    config = LSMConfig(
        ingest=ingest_config,
        query=query_config,
        llm=llm_config,
        vectordb=vectordb_config,
        global_settings=global_config,
        modes=modes,
        notes=notes_config,
        chats=chats_config,
        agents=agents_config,
        remote_providers=remote_providers,
        remote_provider_chains=remote_provider_chains,
        config_path=path,
    )
    config.validate()
    return config


def build_vectordb_config(raw: Dict[str, Any]) -> VectorDBConfig:
    """
    Build VectorDBConfig from raw configuration.
    """
    vectordb_raw = raw.get("vectordb", {})

    return VectorDBConfig(
        provider=vectordb_raw.get("provider", VectorDBConfig.provider),
        collection=vectordb_raw.get("collection", VectorDBConfig.collection),
        persist_dir=vectordb_raw.get("persist_dir", VectorDBConfig.persist_dir),
        chroma_hnsw_space=vectordb_raw.get("chroma_hnsw_space", VectorDBConfig.chroma_hnsw_space),
        connection_string=vectordb_raw.get("connection_string"),
        host=vectordb_raw.get("host"),
        port=vectordb_raw.get("port"),
        database=vectordb_raw.get("database"),
        user=vectordb_raw.get("user"),
        password=vectordb_raw.get("password"),
        index_type=vectordb_raw.get("index_type", "hnsw"),
        pool_size=int(vectordb_raw.get("pool_size", 5)),
    )


def build_query_config(raw: Dict[str, Any]) -> QueryConfig:
    """
    Build QueryConfig from raw configuration.

    Args:
        raw: Raw config dictionary

    Returns:
        QueryConfig instance
    """
    query_section = raw.get("query", {})

    config = QueryConfig(
        k=int(query_section.get("k", QueryConfig.k)),
        retrieve_k=query_section.get("retrieve_k"),
        min_relevance=float(query_section.get("min_relevance", QueryConfig.min_relevance)),
        k_rerank=int(query_section.get("k_rerank", QueryConfig.k_rerank)),
        rerank_strategy=query_section.get("rerank_strategy", "hybrid"),
        no_rerank=bool(query_section.get("no_rerank", False)),
        local_pool=query_section.get("local_pool"),
        max_per_file=int(query_section.get("max_per_file", QueryConfig.max_per_file)),
        mode=query_section.get("mode", "grounded"),
        path_contains=query_section.get("path_contains"),
        ext_allow=query_section.get("ext_allow"),
        ext_deny=query_section.get("ext_deny"),
        enable_query_cache=bool(query_section.get("enable_query_cache", False)),
        query_cache_ttl=int(query_section.get("query_cache_ttl", 3600)),
        query_cache_size=int(query_section.get("query_cache_size", 100)),
        chat_mode=str(query_section.get("chat_mode", "single")),
        enable_llm_server_cache=bool(query_section.get("enable_llm_server_cache", True)),
    )

    return config


def build_notes_config(raw: Dict[str, Any]) -> NotesConfig:
    """
    Build NotesConfig from raw configuration.

    Args:
        raw: Raw notes config dictionary

    Returns:
        NotesConfig instance
    """
    return NotesConfig(
        enabled=bool(raw.get("enabled", True)),
        dir=raw.get("dir", "notes"),
        template=raw.get("template", "default"),
        filename_format=raw.get("filename_format", "timestamp"),
        integration=raw.get("integration", "none"),
        wikilinks=bool(raw.get("wikilinks", False)),
        backlinks=bool(raw.get("backlinks", False)),
        include_tags=bool(raw.get("include_tags", False)),
    )


def build_chats_config(raw: Dict[str, Any]) -> ChatsConfig:
    """
    Build ChatsConfig from raw configuration.

    Args:
        raw: Raw chats config dictionary

    Returns:
        ChatsConfig instance
    """
    return ChatsConfig(
        enabled=bool(raw.get("enabled", True)),
        dir=str(raw.get("dir", "Chats")),
        auto_save=bool(raw.get("auto_save", True)),
        format=str(raw.get("format", "markdown")),
    )


def build_sandbox_config(raw: Dict[str, Any]) -> SandboxConfig:
    """
    Build SandboxConfig from raw configuration.

    Args:
        raw: Raw sandbox config dictionary

    Returns:
        SandboxConfig instance
    """
    if not isinstance(raw, dict):
        raw = {}
    return SandboxConfig(
        allowed_read_paths=[Path(p) for p in raw.get("allowed_read_paths", [])],
        allowed_write_paths=[Path(p) for p in raw.get("allowed_write_paths", [])],
        allow_url_access=bool(raw.get("allow_url_access", False)),
        require_user_permission=dict(raw.get("require_user_permission", {})),
        require_permission_by_risk=dict(raw.get("require_permission_by_risk", {})),
        execution_mode=str(raw.get("execution_mode", "local_only")),
        force_docker=bool(raw.get("force_docker", False)),
        limits=dict(raw.get("limits", {})),
        docker=dict(raw.get("docker", {})),
        tool_llm_assignments=dict(raw.get("tool_llm_assignments", {})),
    )


def build_memory_config(raw: Dict[str, Any]) -> MemoryConfig:
    """
    Build MemoryConfig from raw configuration.

    Args:
        raw: Raw memory config dictionary.

    Returns:
        MemoryConfig instance.
    """
    if not isinstance(raw, dict):
        raw = {}
    return MemoryConfig(
        enabled=bool(raw.get("enabled", True)),
        storage_backend=str(raw.get("storage_backend", "auto")),
        sqlite_path=Path(raw.get("sqlite_path", "memory.sqlite3")),
        postgres_connection_string=raw.get("postgres_connection_string"),
        postgres_table_prefix=str(raw.get("postgres_table_prefix", "agent_memory")),
        ttl_project_fact_days=int(raw.get("ttl_project_fact_days", 90)),
        ttl_task_state_days=int(raw.get("ttl_task_state_days", 7)),
        ttl_cache_hours=int(raw.get("ttl_cache_hours", 24)),
    )


def build_schedule_config(raw: Dict[str, Any]) -> ScheduleConfig:
    """
    Build ScheduleConfig from raw configuration.

    Args:
        raw: Raw schedule config dictionary.

    Returns:
        ScheduleConfig instance.

    Raises:
        ValueError: If schedule config has invalid structure.
    """
    if not isinstance(raw, dict):
        raise ValueError("schedule entry must be an object")
    params = raw.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("schedule params must be an object")
    return ScheduleConfig(
        agent_name=str(raw.get("agent_name", "")),
        params=dict(params),
        interval=str(raw.get("interval", "daily")),
        enabled=bool(raw.get("enabled", True)),
        concurrency_policy=str(raw.get("concurrency_policy", "skip")),
        confirmation_mode=str(raw.get("confirmation_mode", "auto")),
    )


def build_schedule_configs(raw: Any) -> list[ScheduleConfig]:
    """
    Build list of ScheduleConfig from raw configuration.

    Args:
        raw: Raw schedules value.

    Returns:
        Parsed schedule list.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.warn("agents.schedules must be a list. Ignoring invalid value.")
        return []

    schedules: list[ScheduleConfig] = []
    for idx, schedule_raw in enumerate(raw):
        try:
            if isinstance(schedule_raw, ScheduleConfig):
                schedules.append(schedule_raw)
                continue
            schedules.append(build_schedule_config(schedule_raw))
        except Exception as exc:
            warnings.warn(f"Skipping agents.schedules[{idx}] due to error: {exc}")
    return schedules


def build_agent_config(raw: Dict[str, Any] | None) -> AgentConfig | None:
    """
    Build AgentConfig from raw configuration.

    Args:
        raw: Raw agents config dictionary or None

    Returns:
        AgentConfig if present, else None.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        warnings.warn("Config 'agents' must be an object. Ignoring invalid value.")
        return None

    return AgentConfig(
        enabled=bool(raw.get("enabled", False)),
        agents_folder=Path(raw.get("agents_folder", "Agents")),
        max_tokens_budget=int(raw.get("max_tokens_budget", 200_000)),
        max_iterations=int(raw.get("max_iterations", 25)),
        context_window_strategy=str(raw.get("context_window_strategy", "compact")),
        sandbox=build_sandbox_config(raw.get("sandbox", {})),
        memory=build_memory_config(raw.get("memory", {})),
        agent_configs=dict(raw.get("agent_configs", {})),
        schedules=build_schedule_configs(raw.get("schedules")),
    )


def build_source_policy_config(raw: Dict[str, Any]) -> SourcePolicyConfig:
    """
    Build SourcePolicyConfig from raw configuration.

    Args:
        raw: Raw source_policy dictionary

    Returns:
        SourcePolicyConfig instance
    """
    local_raw = raw.get("local", {})
    remote_raw = raw.get("remote", {})
    model_raw = raw.get("model_knowledge", {})

    remote_providers = remote_raw.get("remote_providers")
    if remote_providers is not None and not isinstance(remote_providers, list):
        warnings.warn("Mode remote.remote_providers must be a list of names or {source, weight} objects.")
        remote_providers = None
    if isinstance(remote_providers, list):
        cleaned: list[str | RemoteProviderRef] = []
        for idx, value in enumerate(remote_providers):
            if isinstance(value, str):
                if value.strip():
                    cleaned.append(value.strip())
                continue
            if isinstance(value, dict):
                source = str(value.get("source", "")).strip()
                if not source:
                    warnings.warn(
                        f"Skipping remote.remote_providers[{idx}] because it is missing 'source'."
                    )
                    continue
                weight = value.get("weight")
                if weight is None:
                    cleaned.append(RemoteProviderRef(source=source))
                    continue
                try:
                    cleaned.append(RemoteProviderRef(source=source, weight=float(weight)))
                except (TypeError, ValueError):
                    warnings.warn(
                        f"Skipping remote.remote_providers[{idx}] because weight is invalid."
                    )
                continue
            if isinstance(value, RemoteProviderRef):
                cleaned.append(value)
                continue
            warnings.warn(
                f"Skipping remote.remote_providers[{idx}] because it is not a string or object."
            )
            continue
        remote_providers = cleaned or None

    return SourcePolicyConfig(
        local=LocalSourcePolicy(
            enabled=bool(local_raw.get("enabled", True)),
            min_relevance=float(local_raw.get("min_relevance", 0.25)),
            k=int(local_raw.get("k", 12)),
            k_rerank=int(local_raw.get("k_rerank", 6)),
        ),
        remote=RemoteSourcePolicy(
            enabled=bool(remote_raw.get("enabled", False)),
            rank_strategy=remote_raw.get("rank_strategy", "weighted"),
            max_results=int(remote_raw.get("max_results", 5)),
            remote_providers=remote_providers,
        ),
        model_knowledge=ModelKnowledgePolicy(
            enabled=bool(model_raw.get("enabled", False)),
            require_label=bool(model_raw.get("require_label", True)),
        ),
    )


def build_mode_config(raw: Dict[str, Any]) -> ModeConfig:
    """
    Build ModeConfig from raw configuration.

    Args:
        raw: Raw mode config dictionary

    Returns:
        ModeConfig instance
    """
    source_policy = build_source_policy_config(raw.get("source_policy", {}))

    chats_raw = raw.get("chats")
    chats_config = None
    if isinstance(chats_raw, dict):
        chats_config = ModeChatsConfig(
            auto_save=(
                bool(chats_raw["auto_save"])
                if "auto_save" in chats_raw
                else None
            ),
            dir=chats_raw.get("dir"),
        )

    return ModeConfig(
        synthesis_style=raw.get("synthesis_style", "grounded"),
        source_policy=source_policy,
        chats=chats_config,
    )


def build_modes_registry(raw: Dict[str, Any]) -> Dict[str, ModeConfig]:
    """
    Build modes registry from raw configuration.

    Args:
        raw: Raw config dictionary with 'modes' section

    Returns:
        Dictionary mapping mode names to ModeConfig instances
    """
    modes_raw = raw.get("modes", [])

    if not modes_raw:
        # Return None to use built-in defaults
        return None

    if not isinstance(modes_raw, list):
        warnings.warn("Config 'modes' must be a list. Falling back to built-in modes.")
        return None

    modes_registry: Dict[str, ModeConfig] = {}
    for idx, mode_raw in enumerate(modes_raw):
        if not isinstance(mode_raw, dict):
            warnings.warn(f"Skipping modes[{idx}] because it is not an object.")
            continue
        mode_name = mode_raw.get("name")
        if not mode_name:
            warnings.warn(f"Skipping modes[{idx}] because it is missing 'name'.")
            continue
        if mode_name in modes_registry:
            warnings.warn(f"Skipping duplicate mode name: {mode_name}.")
            continue
        try:
            modes_registry[mode_name] = build_mode_config(mode_raw)
        except Exception as exc:
            warnings.warn(f"Skipping mode '{mode_name}' due to error: {exc}")

    if not modes_registry:
        warnings.warn("No valid modes found. Falling back to built-in modes.")
        return None

    return modes_registry


def build_remote_provider_config(raw: Dict[str, Any]) -> RemoteProviderConfig:
    """
    Build RemoteProviderConfig from raw configuration.

    Args:
        raw: Raw provider config dictionary

    Returns:
        RemoteProviderConfig instance
    """
    extra = {
        key: value for key, value in raw.items() if key not in _REMOTE_PROVIDER_BASE_KEYS
    }
    return RemoteProviderConfig(
        name=raw["name"],  # Required field
        type=raw["type"],  # Required field
        weight=float(raw.get("weight", 1.0)),
        api_key=raw.get("api_key"),
        endpoint=raw.get("endpoint"),
        max_results=raw.get("max_results"),
        language=raw.get("language"),
        user_agent=raw.get("user_agent"),
        timeout=raw.get("timeout"),
        min_interval_seconds=raw.get("min_interval_seconds"),
        section_limit=raw.get("section_limit"),
        snippet_max_chars=raw.get("snippet_max_chars"),
        include_disambiguation=raw.get("include_disambiguation"),
        cache_results=bool(raw.get("cache_results", False)),
        cache_ttl=int(raw.get("cache_ttl", 86400)),
        extra=extra,
    )


def build_remote_providers_registry(raw: Dict[str, Any]) -> list[RemoteProviderConfig]:
    """
    Build remote providers registry from raw configuration.

    Args:
        raw: Raw config dictionary with 'remote_providers' section

    Returns:
        Dictionary mapping provider names to RemoteProviderConfig instances
    """
    providers_raw = raw.get("remote_providers", [])

    if not providers_raw:
        return None

    if not isinstance(providers_raw, list):
        warnings.warn("Config 'remote_providers' must be a list. Ignoring invalid value.")
        return None

    providers: list[RemoteProviderConfig] = []
    for idx, provider_raw in enumerate(providers_raw):
        if not isinstance(provider_raw, dict):
            warnings.warn(f"Skipping remote_providers[{idx}] because it is not an object.")
            continue
        if not provider_raw.get("name"):
            warnings.warn(f"Skipping remote_providers[{idx}] because it is missing 'name'.")
            continue
        if not provider_raw.get("type"):
            warnings.warn(f"Skipping remote_providers[{idx}] because it is missing 'type'.")
            continue
        try:
            providers.append(build_remote_provider_config(provider_raw))
        except Exception as exc:
            warnings.warn(f"Skipping remote provider '{provider_raw.get('name')}' due to error: {exc}")

    if not providers:
        warnings.warn("No valid remote providers found. Remote sources will be disabled.")
        return None

    return providers


def build_chain_link(raw: Dict[str, Any]) -> ChainLink:
    """Build ChainLink from raw configuration."""
    map_values = raw.get("map")
    parsed_map = None
    if map_values is not None:
        if isinstance(map_values, list):
            parsed_map = [str(item) for item in map_values if str(item).strip()]
        else:
            raise ValueError("chain link 'map' must be a list of 'output:input' strings")
    return ChainLink(
        source=str(raw.get("source", "")),
        map=parsed_map,
    )


def build_remote_provider_chain_config(raw: Dict[str, Any]) -> RemoteProviderChainConfig:
    """Build RemoteProviderChainConfig from raw configuration."""
    links_raw = raw.get("links")
    if not isinstance(links_raw, list):
        raise ValueError("remote provider chain 'links' must be a list")

    links: list[ChainLink] = []
    for idx, item in enumerate(links_raw):
        if not isinstance(item, dict):
            raise ValueError(f"remote provider chain link[{idx}] must be an object")
        links.append(build_chain_link(item))

    return RemoteProviderChainConfig(
        name=str(raw.get("name", "")),
        agent_description=str(raw.get("agent_description", "")),
        links=links,
    )


def build_remote_provider_chains_registry(raw: Dict[str, Any]) -> list[RemoteProviderChainConfig] | None:
    """Build remote provider chains registry from raw configuration."""
    chains_raw = raw.get("remote_provider_chains", [])
    if not chains_raw:
        return None
    if not isinstance(chains_raw, list):
        warnings.warn("Config 'remote_provider_chains' must be a list. Ignoring invalid value.")
        return None

    chains: list[RemoteProviderChainConfig] = []
    for idx, chain_raw in enumerate(chains_raw):
        if not isinstance(chain_raw, dict):
            warnings.warn(
                f"Skipping remote_provider_chains[{idx}] because it is not an object."
            )
            continue
        if not chain_raw.get("name"):
            warnings.warn(
                f"Skipping remote_provider_chains[{idx}] because it is missing 'name'."
            )
            continue
        try:
            chains.append(build_remote_provider_chain_config(chain_raw))
        except Exception as exc:
            warnings.warn(
                f"Skipping remote provider chain '{chain_raw.get('name')}' due to error: {exc}"
            )

    if not chains:
        warnings.warn("No valid remote provider chains found. Ignoring chain configuration.")
        return None

    return chains


def load_config_from_file(path: Path | str) -> LSMConfig:
    """
    Load and validate LSM configuration from file.

    This is the main entry point for loading configuration.

    Args:
        path: Path to config file (.json, .yaml, or .yml)

    Returns:
        Validated LSMConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid

    Example:
        >>> config = load_config_from_file("config.json")
        >>> print(config.ingest.roots)
        [PosixPath('/path/to/documents')]
    """
    if isinstance(path, str):
        path = Path(path)

    path = path.expanduser().resolve()

    raw = load_raw_config(path)
    return build_config_from_raw(raw, path)


def save_config_to_file(config: LSMConfig, path: Path | str) -> None:
    """
    Serialize and save configuration to JSON/YAML file.

    Args:
        config: LSMConfig instance to save
        path: Destination config file path
    """
    if isinstance(path, str):
        path = Path(path)

    path = path.expanduser().resolve()
    raw = config_to_raw(config)

    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    elif suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported config format: {suffix}. "
            f"Use .json, .yaml, or .yml"
        )


def config_to_raw(config: LSMConfig) -> Dict[str, Any]:
    """
    Serialize LSMConfig into a JSON/YAML-friendly dict.
    """
    providers_raw: list[Dict[str, Any]] = []
    for provider in config.llm.providers:
        providers_raw.append({
            "provider_name": provider.provider_name,
            "api_key": provider.api_key,
            "base_url": provider.base_url,
            "endpoint": provider.endpoint,
            "api_version": provider.api_version,
            "deployment_name": provider.deployment_name,
        })

    services_raw: Dict[str, Dict[str, Any]] = {}
    for name, service in config.llm.services.items():
        entry: Dict[str, Any] = {
            "provider": service.provider,
            "model": service.model,
        }
        if service.temperature is not None:
            entry["temperature"] = service.temperature
        if service.max_tokens is not None:
            entry["max_tokens"] = service.max_tokens
        services_raw[name] = entry

    modes = []
    if config.modes:
        for name, mode in config.modes.items():
            serialized_remote_providers = None
            if mode.source_policy.remote.remote_providers:
                serialized_remote_providers = []
                for item in mode.source_policy.remote.remote_providers:
                    if isinstance(item, RemoteProviderRef):
                        value: Dict[str, Any] = {"source": item.source}
                        if item.weight is not None:
                            value["weight"] = item.weight
                        serialized_remote_providers.append(value)
                    else:
                        serialized_remote_providers.append(item)
            mode_entry = {
                "name": name,
                "synthesis_style": mode.synthesis_style,
                "source_policy": {
                    "local": {
                        "enabled": mode.source_policy.local.enabled,
                        "min_relevance": mode.source_policy.local.min_relevance,
                        "k": mode.source_policy.local.k,
                        "k_rerank": mode.source_policy.local.k_rerank,
                    },
                    "remote": {
                        "enabled": mode.source_policy.remote.enabled,
                        "rank_strategy": mode.source_policy.remote.rank_strategy,
                        "max_results": mode.source_policy.remote.max_results,
                        "remote_providers": serialized_remote_providers,
                    },
                    "model_knowledge": {
                        "enabled": mode.source_policy.model_knowledge.enabled,
                        "require_label": mode.source_policy.model_knowledge.require_label,
                    },
                },
            }
            if mode.chats is not None:
                chats_entry: Dict[str, Any] = {}
                if mode.chats.auto_save is not None:
                    chats_entry["auto_save"] = mode.chats.auto_save
                if mode.chats.dir:
                    chats_entry["dir"] = mode.chats.dir
                if chats_entry:
                    mode_entry["chats"] = chats_entry
            modes.append(mode_entry)

    remote_providers = None
    if config.remote_providers:
        remote_providers = []
        for provider in config.remote_providers:
            provider_entry = {
                "name": provider.name,
                "type": provider.type,
                "weight": provider.weight,
                "api_key": provider.api_key,
                "endpoint": provider.endpoint,
                "max_results": provider.max_results,
                "language": provider.language,
                "user_agent": provider.user_agent,
                "timeout": provider.timeout,
                "min_interval_seconds": provider.min_interval_seconds,
                "section_limit": provider.section_limit,
                "snippet_max_chars": provider.snippet_max_chars,
                "include_disambiguation": provider.include_disambiguation,
                "cache_results": provider.cache_results,
                "cache_ttl": provider.cache_ttl,
            }
            for key, value in (provider.extra or {}).items():
                if key not in provider_entry:
                    provider_entry[key] = value
            remote_providers.append(provider_entry)

    remote_provider_chains = None
    if config.remote_provider_chains:
        remote_provider_chains = []
        for chain in config.remote_provider_chains:
            chain_entry: Dict[str, Any] = {
                "name": chain.name,
                "agent_description": chain.agent_description,
                "links": [],
            }
            for link in chain.links:
                link_entry: Dict[str, Any] = {"source": link.source}
                if link.map:
                    link_entry["map"] = list(link.map)
                chain_entry["links"].append(link_entry)
            remote_provider_chains.append(chain_entry)

    gs = config.global_settings
    agents_raw = None
    if config.agents is not None:
        agents_raw = {
            "enabled": config.agents.enabled,
            "agents_folder": str(config.agents.agents_folder),
            "max_tokens_budget": config.agents.max_tokens_budget,
            "max_iterations": config.agents.max_iterations,
            "context_window_strategy": config.agents.context_window_strategy,
            "sandbox": {
                "allowed_read_paths": [str(p) for p in config.agents.sandbox.allowed_read_paths],
                "allowed_write_paths": [str(p) for p in config.agents.sandbox.allowed_write_paths],
                "allow_url_access": config.agents.sandbox.allow_url_access,
                "require_user_permission": dict(config.agents.sandbox.require_user_permission),
                "require_permission_by_risk": dict(config.agents.sandbox.require_permission_by_risk),
                "execution_mode": config.agents.sandbox.execution_mode,
                "force_docker": config.agents.sandbox.force_docker,
                "limits": dict(config.agents.sandbox.limits),
                "docker": dict(config.agents.sandbox.docker),
                "tool_llm_assignments": dict(config.agents.sandbox.tool_llm_assignments),
            },
            "memory": {
                "enabled": config.agents.memory.enabled,
                "storage_backend": config.agents.memory.storage_backend,
                "sqlite_path": str(config.agents.memory.sqlite_path),
                "postgres_connection_string": config.agents.memory.postgres_connection_string,
                "postgres_table_prefix": config.agents.memory.postgres_table_prefix,
                "ttl_project_fact_days": config.agents.memory.ttl_project_fact_days,
                "ttl_task_state_days": config.agents.memory.ttl_task_state_days,
                "ttl_cache_hours": config.agents.memory.ttl_cache_hours,
            },
            "agent_configs": dict(config.agents.agent_configs),
            "schedules": [
                {
                    "agent_name": schedule.agent_name,
                    "params": dict(schedule.params),
                    "interval": schedule.interval,
                    "enabled": schedule.enabled,
                    "concurrency_policy": schedule.concurrency_policy,
                    "confirmation_mode": schedule.confirmation_mode,
                }
                for schedule in config.agents.schedules
            ],
        }
    return {
        "global": {
            "global_folder": str(gs.global_folder) if gs.global_folder else None,
            "embed_model": gs.embed_model,
            "device": gs.device,
            "batch_size": gs.batch_size,
            "embedding_dimension": gs.embedding_dimension,
        },
        "ingest": {
            "roots": [
                (
                    {"path": str(rc.path), "tags": rc.tags, "content_type": rc.content_type}
                    if rc.tags or rc.content_type
                    else str(rc.path)
                )
                for rc in config.ingest.roots
            ],
            "persist_dir": str(config.ingest.persist_dir),
            "collection": config.ingest.collection,
            "chroma_flush_interval": config.ingest.chroma_flush_interval,
            "manifest": str(config.ingest.manifest),
            "chunk_size": config.ingest.chunk_size,
            "chunk_overlap": config.ingest.chunk_overlap,
            "chunking_strategy": config.ingest.chunking_strategy,
            "enable_ocr": config.ingest.enable_ocr,
            "enable_ai_tagging": config.ingest.enable_ai_tagging,
            "tags_per_chunk": config.ingest.tags_per_chunk,
            "dry_run": config.ingest.dry_run,
            "skip_errors": config.ingest.skip_errors,
            "enable_language_detection": config.ingest.enable_language_detection,
            "enable_translation": config.ingest.enable_translation,
            "translation_target": config.ingest.translation_target,
            "extensions": config.ingest.extensions,
            "override_extensions": config.ingest.override_extensions,
            "exclude_dirs": config.ingest.exclude_dirs,
            "override_excludes": config.ingest.override_excludes,
            "max_files": config.ingest.max_files,
            "max_seconds": config.ingest.max_seconds,
            "enable_versioning": config.ingest.enable_versioning,
        },
        "llms": {
            "providers": providers_raw,
            "services": services_raw,
        },
        "query": {
            "mode": config.query.mode,
            "k": config.query.k,
            "retrieve_k": config.query.retrieve_k,
            "min_relevance": config.query.min_relevance,
            "k_rerank": config.query.k_rerank,
            "rerank_strategy": config.query.rerank_strategy,
            "no_rerank": config.query.no_rerank,
            "local_pool": config.query.local_pool,
            "max_per_file": config.query.max_per_file,
            "path_contains": config.query.path_contains,
            "ext_allow": config.query.ext_allow,
            "ext_deny": config.query.ext_deny,
            "enable_query_cache": config.query.enable_query_cache,
            "query_cache_ttl": config.query.query_cache_ttl,
            "query_cache_size": config.query.query_cache_size,
            "chat_mode": config.query.chat_mode,
            "enable_llm_server_cache": config.query.enable_llm_server_cache,
        },
        "modes": modes or None,
        "notes": {
            "enabled": config.notes.enabled,
            "dir": config.notes.dir,
            "template": config.notes.template,
            "filename_format": config.notes.filename_format,
            "integration": config.notes.integration,
            "wikilinks": config.notes.wikilinks,
            "backlinks": config.notes.backlinks,
            "include_tags": config.notes.include_tags,
        },
        "chats": {
            "enabled": config.chats.enabled,
            "dir": config.chats.dir,
            "auto_save": config.chats.auto_save,
            "format": config.chats.format,
        },
        "agents": agents_raw,
        "remote_providers": remote_providers,
        "remote_provider_chains": remote_provider_chains,
        "vectordb": {
            "provider": config.vectordb.provider,
            "collection": config.vectordb.collection,
            "persist_dir": str(config.vectordb.persist_dir),
            "chroma_hnsw_space": config.vectordb.chroma_hnsw_space,
            "connection_string": config.vectordb.connection_string,
            "host": config.vectordb.host,
            "port": config.vectordb.port,
            "database": config.vectordb.database,
            "user": config.vectordb.user,
            "password": config.vectordb.password,
            "index_type": config.vectordb.index_type,
            "pool_size": config.vectordb.pool_size,
        },
    }
