"""
Agent tool framework and built-in tools.
"""

from typing import Optional, TYPE_CHECKING

from .base import BaseTool, ToolRegistry
from .sandbox import ToolSandbox
from .read_file import ReadFileTool
from .read_folder import ReadFolderTool
from .write_file import WriteFileTool
from .append_file import AppendFileTool
from .create_folder import CreateFolderTool
from .extract_snippets import ExtractSnippetsTool
from .file_metadata import FileMetadataTool
from .hash_file import HashFileTool
from .load_url import LoadURLTool
from .query_embeddings import QueryEmbeddingsTool
from .query_llm import QueryLLMTool
from .query_remote import QueryRemoteTool
from .query_remote_chain import QueryRemoteChainTool
from .similarity_search import SimilaritySearchTool
from .source_map import SourceMapTool
from .memory_put import MemoryPutTool
from .memory_search import MemorySearchTool
from .docker_runner import DockerRunner
from .runner import BaseRunner, LocalRunner, ToolExecutionResult

if TYPE_CHECKING:
    from lsm.config.models import LSMConfig
    from lsm.vectordb.base import BaseVectorDBProvider
    from lsm.agents.memory import BaseMemoryStore

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolSandbox",
    "ReadFileTool",
    "ReadFolderTool",
    "WriteFileTool",
    "AppendFileTool",
    "CreateFolderTool",
    "ExtractSnippetsTool",
    "FileMetadataTool",
    "HashFileTool",
    "LoadURLTool",
    "QueryEmbeddingsTool",
    "QueryLLMTool",
    "QueryRemoteTool",
    "QueryRemoteChainTool",
    "SimilaritySearchTool",
    "SourceMapTool",
    "MemoryPutTool",
    "MemorySearchTool",
    "DockerRunner",
    "BaseRunner",
    "LocalRunner",
    "ToolExecutionResult",
    "create_default_tool_registry",
]


def create_default_tool_registry(
    config: "LSMConfig",
    *,
    collection: Optional["BaseVectorDBProvider"] = None,
    embedder=None,
    batch_size: int = 32,
    memory_store: Optional["BaseMemoryStore"] = None,
) -> ToolRegistry:
    """
    Build a tool registry with built-in default tools.

    Args:
        config: Loaded LSM configuration.
        collection: Optional vector DB provider for embedding queries.
        embedder: Optional embedding model instance.
        batch_size: Embedding batch size for query tool.
        memory_store: Optional pre-initialized memory store backend.

    Returns:
        Populated ToolRegistry instance.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ReadFolderTool())
    registry.register(FileMetadataTool())
    registry.register(HashFileTool())
    registry.register(SourceMapTool())
    registry.register(WriteFileTool())
    registry.register(AppendFileTool())
    registry.register(CreateFolderTool())
    registry.register(LoadURLTool())
    registry.register(QueryLLMTool(config.llm))
    registry.register(QueryRemoteTool(config))
    registry.register(QueryRemoteChainTool(config))
    if collection is not None:
        registry.register(SimilaritySearchTool(collection=collection))
    if collection is not None and embedder is not None:
        registry.register(
            QueryEmbeddingsTool(
                collection=collection,
                embedder=embedder,
                batch_size=batch_size,
            )
        )
        registry.register(
            ExtractSnippetsTool(
                collection=collection,
                embedder=embedder,
                batch_size=batch_size,
            )
        )
    agents_cfg = getattr(config, "agents", None)
    memory_enabled = bool(
        agents_cfg
        and getattr(agents_cfg, "enabled", False)
        and getattr(getattr(agents_cfg, "memory", None), "enabled", False)
    )
    if memory_enabled and memory_store is not None:
        registry.register(MemoryPutTool(memory_store))
        registry.register(MemorySearchTool(memory_store))
    return registry
