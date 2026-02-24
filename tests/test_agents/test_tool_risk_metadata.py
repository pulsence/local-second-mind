from __future__ import annotations

import pytest

from lsm.agents.tools import (
    AskUserTool,
    AwaitAgentTool,
    AppendFileTool,
    BashTool,
    CollectArtifactsTool,
    CreateFolderTool,
    EditFileTool,
    ExtractSnippetsTool,
    FileMetadataTool,
    FindFileTool,
    FindSectionTool,
    HashFileTool,
    LoadURLTool,
    MemoryPutTool,
    MemoryRemoveTool,
    MemorySearchTool,
    PowerShellTool,
    QueryEmbeddingsTool,
    QueryLLMTool,
    QueryRemoteChainTool,
    QueryRemoteTool,
    ReadFileTool,
    ReadFolderTool,
    SimilaritySearchTool,
    SourceMapTool,
    SpawnAgentTool,
    WriteFileTool,
)
from lsm.agents.tools.base import BaseTool, ToolRegistry


class DefaultTool(BaseTool):
    name = "default_tool"
    description = "Default metadata tool."

    def execute(self, args: dict) -> str:
        return "ok"


class NetworkRiskNoFlagTool(BaseTool):
    name = "network_risk_no_flag"
    description = "Network risk without needs_network flag."
    risk_level = "network"

    def execute(self, args: dict) -> str:
        return "ok"


def test_base_tool_definition_includes_risk_metadata_defaults() -> None:
    definition = DefaultTool().get_definition()
    assert definition["risk_level"] == "read_only"
    assert definition["preferred_runner"] == "local"
    assert definition["needs_network"] is False


@pytest.mark.parametrize(
    ("tool_cls", "risk_level", "needs_network"),
    [
        (ReadFileTool, "read_only", False),
        (ReadFolderTool, "read_only", False),
        (FileMetadataTool, "read_only", False),
        (HashFileTool, "read_only", False),
        (FindFileTool, "read_only", False),
        (FindSectionTool, "read_only", False),
        (ExtractSnippetsTool, "read_only", False),
        (SimilaritySearchTool, "read_only", False),
        (SourceMapTool, "read_only", False),
        (AskUserTool, "read_only", False),
        (MemorySearchTool, "read_only", False),
        (QueryEmbeddingsTool, "read_only", False),
        (MemoryPutTool, "writes_workspace", False),
        (MemoryRemoveTool, "writes_workspace", False),
        (WriteFileTool, "writes_workspace", False),
        (AppendFileTool, "writes_workspace", False),
        (CreateFolderTool, "writes_workspace", False),
        (EditFileTool, "writes_workspace", False),
        (LoadURLTool, "network", True),
        (QueryLLMTool, "network", True),
        (QueryRemoteTool, "network", True),
        (QueryRemoteChainTool, "network", True),
        (SpawnAgentTool, "exec", False),
        (AwaitAgentTool, "exec", False),
        (CollectArtifactsTool, "exec", False),
        (BashTool, "exec", False),
        (PowerShellTool, "exec", False),
    ],
)
def test_builtin_tool_classes_have_expected_risk_metadata(
    tool_cls: type[BaseTool],
    risk_level: str,
    needs_network: bool,
) -> None:
    assert tool_cls.risk_level == risk_level
    assert tool_cls.needs_network is needs_network
    assert tool_cls.preferred_runner == "local"


def test_tool_registry_filters_by_risk_level() -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(LoadURLTool())

    assert [tool.name for tool in registry.list_by_risk("read_only")] == ["read_file"]
    assert [tool.name for tool in registry.list_by_risk("writes_workspace")] == ["write_file"]
    assert [tool.name for tool in registry.list_by_risk("network")] == ["load_url"]


def test_tool_registry_lists_network_tools() -> None:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(LoadURLTool())
    registry.register(NetworkRiskNoFlagTool())

    assert [tool.name for tool in registry.list_network_tools()] == [
        "load_url",
        "network_risk_no_flag",
    ]
