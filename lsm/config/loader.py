"""
Configuration loader for Local Second Mind.

Handles loading configuration from JSON/YAML files and converting
to typed dataclass models.
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

from .models import (
    LSMConfig,
    IngestConfig,
    QueryConfig,
    LLMConfig,
    FeatureLLMConfig,
    ModeConfig,
    SourcePolicyConfig,
    LocalSourcePolicy,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    NotesConfig,
    RemoteProviderConfig,
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
        provider=raw.get("provider"),
        model=raw.get("model"),
        api_key=raw.get("api_key"),
        temperature=raw.get("temperature"),
        max_tokens=raw.get("max_tokens"),
    )


def build_llm_config(raw: Dict[str, Any]) -> LLMConfig:
    """
    Build LLMConfig from raw configuration.

    Args:
        raw: Raw config dictionary

    Returns:
        LLMConfig instance
    """
    # Check for OpenAI section (legacy) or llm section (new)
    llm_config = raw.get("llm", {})
    openai_config = raw.get("openai", {})

    # Merge configurations (llm takes precedence)
    config = {
        "provider": llm_config.get("provider", "openai"),
        "model": llm_config.get("model") or raw.get("query", {}).get("model", "gpt-5.2"),
        "api_key": llm_config.get("api_key") or openai_config.get("api_key"),
        "temperature": llm_config.get("temperature", 0.7),
        "max_tokens": llm_config.get("max_tokens", 2000),
    }

    # Build per-feature overrides if present
    if "query" in llm_config:
        config["query"] = build_feature_llm_config(llm_config["query"])

    if "tagging" in llm_config:
        config["tagging"] = build_feature_llm_config(llm_config["tagging"])

    if "ranking" in llm_config:
        config["ranking"] = build_feature_llm_config(llm_config["ranking"])

    return LLMConfig(**config)


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

    # Build config
    config = IngestConfig(
        roots=[Path(r) for r in roots],
        embed_model=raw.get("embed_model", IngestConfig.embed_model),
        device=raw.get("device", IngestConfig.device),
        batch_size=int(raw.get("batch_size", IngestConfig.batch_size)),
        persist_dir=Path(raw.get("persist_dir", ".chroma")),
        collection=raw.get("collection", IngestConfig.collection),
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
    modes_raw = raw.get("modes", {})

    if not modes_raw:
        # Return None to use built-in defaults
        return None

    return {
        mode_name: build_mode_config(mode_raw)
        for mode_name, mode_raw in modes_raw.items()
    }


def build_remote_provider_config(raw: Dict[str, Any]) -> RemoteProviderConfig:
    """
    Build RemoteProviderConfig from raw configuration.

    Args:
        raw: Raw provider config dictionary

    Returns:
        RemoteProviderConfig instance
    """
    return RemoteProviderConfig(
        type=raw["type"],  # Required field
        enabled=bool(raw.get("enabled", True)),
        weight=float(raw.get("weight", 1.0)),
        api_key=raw.get("api_key"),
        endpoint=raw.get("endpoint"),
        max_results=raw.get("max_results"),
    )


def build_remote_providers_registry(raw: Dict[str, Any]) -> Dict[str, RemoteProviderConfig]:
    """
    Build remote providers registry from raw configuration.

    Args:
        raw: Raw config dictionary with 'remote_providers' section

    Returns:
        Dictionary mapping provider names to RemoteProviderConfig instances
    """
    providers_raw = raw.get("remote_providers", {})

    if not providers_raw:
        return None

    return {
        provider_name: build_remote_provider_config(provider_raw)
        for provider_name, provider_raw in providers_raw.items()
    }


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

    # Build mode system configs
    modes = build_modes_registry(raw)
    remote_providers = build_remote_providers_registry(raw)

    # Build top-level config
    config = LSMConfig(
        ingest=ingest_config,
        query=query_config,
        llm=llm_config,
        modes=modes,
        remote_providers=remote_providers,
        config_path=path,
    )

    # Validate
    config.validate()

    return config


