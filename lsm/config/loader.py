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
    IngestConfig,
    QueryConfig,
    LLMRegistryConfig,
    LLMProviderConfig,
    FeatureLLMConfig,
    ModeConfig,
    SourcePolicyConfig,
    LocalSourcePolicy,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    NotesConfig,
    RemoteProviderConfig,
    VectorDBConfig,
    DEFAULT_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
)


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


def build_feature_llm_config(raw: Dict[str, Any]) -> FeatureLLMConfig:
    """
    Build FeatureLLMConfig from raw configuration.

    Args:
        raw: Raw config dictionary for a feature override

    Returns:
        FeatureLLMConfig instance
    """
    return FeatureLLMConfig(
        model=raw.get("model"),
        api_key=raw.get("api_key"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        base_url=raw.get("base_url"),
        endpoint=raw.get("endpoint"),
        api_version=raw.get("api_version"),
        deployment_name=raw.get("deployment_name"),
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
        raise ValueError("Each llms[] entry must include 'provider_name'")

    return LLMProviderConfig(
        provider_name=provider_name,
        api_key=raw.get("api_key"),
        model=raw.get("model"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
        base_url=raw.get("base_url"),
        endpoint=raw.get("endpoint"),
        api_version=raw.get("api_version"),
        deployment_name=raw.get("deployment_name"),
        query=build_feature_llm_config(raw["query"]) if "query" in raw else None,
        tagging=build_feature_llm_config(raw["tagging"]) if "tagging" in raw else None,
        ranking=build_feature_llm_config(raw["ranking"]) if "ranking" in raw else None,
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
    if not llms_raw or not isinstance(llms_raw, list):
        raise ValueError("Config must include 'llms' as a non-empty list")

    providers = [build_llm_provider_config(item) for item in llms_raw]
    return LLMRegistryConfig(llms=providers)


def build_ingest_config(raw: Dict[str, Any], config_path: Path) -> IngestConfig:
    """
    Build IngestConfig from raw configuration.

    Args:
        raw: Raw config dictionary
        config_path: Path to config file (for resolving relative paths)

    Returns:
        IngestConfig instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Required field
    roots = raw.get("roots")
    if not roots or not isinstance(roots, list):
        raise ValueError("Config must include 'roots' as a non-empty list of directory paths")

    vectordb_raw = raw.get("vectordb", {})
    persist_dir = vectordb_raw.get("persist_dir", raw.get("persist_dir", ".chroma"))
    collection = vectordb_raw.get("collection", raw.get("collection", IngestConfig.collection))

    # Build config
    config = IngestConfig(
        roots=[Path(r) for r in roots],
        embed_model=raw.get("embed_model", IngestConfig.embed_model),
        device=raw.get("device", IngestConfig.device),
        batch_size=int(raw.get("batch_size", IngestConfig.batch_size)),
        persist_dir=Path(persist_dir),
        collection=collection,
        chroma_flush_interval=int(raw.get("chroma_flush_interval", IngestConfig.chroma_flush_interval)),
        manifest=Path(raw.get("manifest", ".ingest/manifest.json")),
        extensions=raw.get("extensions"),
        override_extensions=bool(raw.get("override_extensions", False)),
        exclude_dirs=raw.get("exclude_dirs"),
        override_excludes=bool(raw.get("override_excludes", False)),
        chunk_size=int(raw.get("chunk_size", IngestConfig.chunk_size)),
        chunk_overlap=int(raw.get("chunk_overlap", IngestConfig.chunk_overlap)),
        enable_ocr=bool(raw.get("enable_ocr", False)),
        enable_ai_tagging=bool(raw.get("enable_ai_tagging", False)),
        tagging_model=raw.get("tagging_model", "gpt-5.2"),
        tags_per_chunk=int(raw.get("tags_per_chunk", 3)),
        dry_run=bool(raw.get("dry_run", False)),
        skip_errors=bool(raw.get("skip_errors", True)),
    )

    return config


def build_vectordb_config(raw: Dict[str, Any]) -> VectorDBConfig:
    """
    Build VectorDBConfig from raw configuration.
    """
    vectordb_raw = raw.get("vectordb", {})

    return VectorDBConfig(
        provider=vectordb_raw.get("provider", VectorDBConfig.provider),
        collection=vectordb_raw.get("collection", raw.get("collection", VectorDBConfig.collection)),
        persist_dir=vectordb_raw.get("persist_dir", raw.get("persist_dir", VectorDBConfig.persist_dir)),
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

    return SourcePolicyConfig(
        local=LocalSourcePolicy(
            min_relevance=float(local_raw.get("min_relevance", 0.25)),
            k=int(local_raw.get("k", 12)),
            k_rerank=int(local_raw.get("k_rerank", 6)),
        ),
        remote=RemoteSourcePolicy(
            enabled=bool(remote_raw.get("enabled", False)),
            rank_strategy=remote_raw.get("rank_strategy", "weighted"),
            max_results=int(remote_raw.get("max_results", 5)),
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
    notes = build_notes_config(raw.get("notes", {}))

    return ModeConfig(
        synthesis_style=raw.get("synthesis_style", "grounded"),
        source_policy=source_policy,
        notes=notes,
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
    return RemoteProviderConfig(
        name=raw["name"],  # Required field
        type=raw["type"],  # Required field
        enabled=bool(raw.get("enabled", True)),
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

    # Load raw config
    raw = load_raw_config(path)

    # Build component configs
    llm_config = build_llm_config(raw)
    ingest_config = build_ingest_config(raw, path)
    query_config = build_query_config(raw)
    vectordb_config = build_vectordb_config(raw)

    # Build mode system configs
    modes = build_modes_registry(raw)
    remote_providers = build_remote_providers_registry(raw)

    # Build top-level config
    config = LSMConfig(
        ingest=ingest_config,
        query=query_config,
        llm=llm_config,
        vectordb=vectordb_config,
        modes=modes,
        remote_providers=remote_providers,
        config_path=path,
    )

    # Validate
    config.validate()

    return config


