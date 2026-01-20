"""
Configuration data models for Local Second Mind.

This package provides typed dataclasses for configuration options and defaults.
"""

from .constants import (
    DEFAULT_EXTENSIONS,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHROMA_FLUSH_INTERVAL,
    DEFAULT_VDB_PROVIDER,
    DEFAULT_CHROMA_HNSW_SPACE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K,
    DEFAULT_K_RERANK,
    DEFAULT_MAX_PER_FILE,
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_MAX_TOKENS,
)
from .llm import FeatureLLMConfig, LLMConfig, LLMProviderConfig, LLMRegistryConfig
from .ingest import IngestConfig
from .vectordb import VectorDBConfig
from .modes import (
    LocalSourcePolicy,
    RemoteProviderRef,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    SourcePolicyConfig,
    NotesConfig,
    ModeConfig,
    RemoteProviderConfig,
)
from .query import QueryConfig
from .lsm_config import LSMConfig

__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFAULT_EXCLUDE_DIRS",
    "DEFAULT_COLLECTION",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_DEVICE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHROMA_FLUSH_INTERVAL",
    "DEFAULT_VDB_PROVIDER",
    "DEFAULT_CHROMA_HNSW_SPACE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_K",
    "DEFAULT_K_RERANK",
    "DEFAULT_MAX_PER_FILE",
    "DEFAULT_MIN_RELEVANCE",
    "DEFAULT_LLM_TEMPERATURE",
    "DEFAULT_LLM_MAX_TOKENS",
    "FeatureLLMConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "LLMRegistryConfig",
    "IngestConfig",
    "VectorDBConfig",
    "LocalSourcePolicy",
    "RemoteProviderRef",
    "RemoteSourcePolicy",
    "ModelKnowledgePolicy",
    "SourcePolicyConfig",
    "NotesConfig",
    "ModeConfig",
    "RemoteProviderConfig",
    "QueryConfig",
    "LSMConfig",
]
