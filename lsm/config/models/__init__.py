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
from .global_config import GlobalConfig, MCPServerConfig
from .llm import (
    LLMConfig,
    LLMProviderConfig,
    LLMRegistryConfig,
    LLMServiceConfig,
    LLMTierConfig,
)
from .ingest import IngestConfig, RootConfig
from .vectordb import VectorDBConfig
from .modes import (
    LocalSourcePolicy,
    RemoteProviderRef,
    RemoteSourcePolicy,
    ModelKnowledgePolicy,
    SourcePolicyConfig,
    NotesConfig,
    ModeChatsConfig,
    ModeConfig,
    RemoteProviderConfig,
    ChainLink,
    RemoteProviderChainConfig,
    RemoteConfig,
)
from .query import QueryConfig
from .chats import ChatsConfig
from .agents import (
    AgentConfig,
    InteractionConfig,
    MemoryConfig,
    SandboxConfig,
    ScheduleConfig,
)
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
    "GlobalConfig",
    "MCPServerConfig",
    "LLMConfig",
    "LLMProviderConfig",
    "LLMRegistryConfig",
    "LLMServiceConfig",
    "LLMTierConfig",
    "IngestConfig",
    "RootConfig",
    "VectorDBConfig",
    "LocalSourcePolicy",
    "RemoteProviderRef",
    "RemoteSourcePolicy",
    "ModelKnowledgePolicy",
    "SourcePolicyConfig",
    "NotesConfig",
    "ModeChatsConfig",
    "ModeConfig",
    "RemoteProviderConfig",
    "ChainLink",
    "RemoteProviderChainConfig",
    "RemoteConfig",
    "QueryConfig",
    "ChatsConfig",
    "AgentConfig",
    "InteractionConfig",
    "MemoryConfig",
    "SandboxConfig",
    "ScheduleConfig",
    "LSMConfig",
]
