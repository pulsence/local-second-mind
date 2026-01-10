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

    # Build top-level config
    config = LSMConfig(
        ingest=ingest_config,
        query=query_config,
        llm=llm_config,
        config_path=path,
    )

    # Validate
    config.validate()

    return config


# Backward compatibility helpers

def load_config_dict(path: Path) -> Dict[str, Any]:
    """
    Load config as dictionary (for backward compatibility).

    This is deprecated. Use load_config_from_file() instead.
    """
    return load_raw_config(path)


def normalize_config_legacy(raw: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """
    Legacy function for backward compatibility.

    Loads config using new system, then converts back to dict.
    This maintains compatibility with old code during migration.
    """
    config = load_config_from_file(config_path)

    # Convert back to legacy dict format
    return {
        # Shared
        "roots": config.ingest.roots,
        "persist_dir": config.persist_dir,
        "collection": config.collection,
        "embed_model": config.embed_model,
        "device": config.device,
        "batch_size": config.batch_size,

        # Ingest
        "chroma_flush_interval": config.ingest.chroma_flush_interval,
        "manifest": config.ingest.manifest,
        "exts": config.ingest.exts,
        "exclude_dirs": config.ingest.exclude_set,
        "dry_run": config.ingest.dry_run,

        # LLM
        "openai_api_key": config.llm.api_key,
        "model": config.llm.model,

        # Query
        "k": config.query.k,
        "k_rerank": config.query.k_rerank,
        "no_rerank": config.query.no_rerank,
        "max_per_file": config.query.max_per_file,
        "local_pool": config.query.local_pool,
        "min_relevance": config.query.min_relevance,
        "path_contains": config.query.path_contains,
        "ext_allow": config.query.ext_allow,
        "ext_deny": config.query.ext_deny,
        "retrieve_k": config.query.retrieve_k,
    }
