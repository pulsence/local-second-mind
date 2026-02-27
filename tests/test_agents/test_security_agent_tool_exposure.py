from __future__ import annotations

from pathlib import Path

import pytest

from lsm.agents.academic import CuratorAgent, ResearchAgent, SynthesisAgent
from lsm.agents.assistants import (
    AssistantAgent,
    CalendarAssistantAgent,
    EmailAssistantAgent,
    NewsAssistantAgent,
)
from lsm.agents.meta import AssistantMetaAgent, MetaAgent
from lsm.agents.productivity import (
    GeneralAgent,
    LibrarianAgent,
    ManuscriptEditorAgent,
    WritingAgent,
)
from lsm.agents.tools.base import BaseTool, ToolRegistry
from lsm.agents.tools.sandbox import ToolSandbox
from lsm.config.loader import build_config_from_raw


class _StubTool(BaseTool):
    description = "Stub tool."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, name: str) -> None:
        self.name = name

    def execute(self, args: dict) -> str:
        return ""


_TOOL_NAMES = {
    "ask_user",
    "read_file",
    "read_folder",
    "find_file",
    "find_section",
    "file_metadata",
    "hash_file",
    "write_file",
    "append_file",
    "create_folder",
    "edit_file",
    "source_map",
    "extract_snippets",
    "similarity_search",
    "query_knowledge_base",
    "query_arxiv",
    "query_remote_chain",
    "query_llm",
    "load_url",
    "memory_put",
    "memory_search",
    "memory_remove",
    "spawn_agent",
    "await_agent",
    "collect_artifacts",
    "bash",
    "powershell",
}

_AGENT_CLASSES = [
    ResearchAgent,
    CuratorAgent,
    SynthesisAgent,
    GeneralAgent,
    WritingAgent,
    LibrarianAgent,
    ManuscriptEditorAgent,
    AssistantAgent,
    CalendarAssistantAgent,
    EmailAssistantAgent,
    NewsAssistantAgent,
    MetaAgent,
    AssistantMetaAgent,
]

_STRIDE_TEST_MODULES = {
    "read_file": {"test_security_paths.py", "test_security_integrity.py"},
    "read_folder": {"test_security_paths.py"},
    "find_file": {"test_security_paths.py"},
    "find_section": {"test_security_paths.py"},
    "file_metadata": {"test_security_paths.py"},
    "hash_file": {"test_security_integrity.py"},
    "write_file": {"test_security_paths.py", "test_security_integrity.py"},
    "append_file": {"test_security_paths.py"},
    "create_folder": {"test_security_paths.py"},
    "edit_file": {"test_security_paths.py"},
    "source_map": {"test_security_integrity.py"},
    "extract_snippets": {"test_security_resources.py"},
    "similarity_search": {"test_security_resources.py"},
    "query_knowledge_base": {"test_security_resources.py"},
    "query_arxiv": {"test_security_network.py"},
    "query_remote_chain": {"test_security_network.py"},
    "query_llm": {"test_security_network.py"},
    "load_url": {"test_security_network.py"},
    "memory_put": {"test_security_integrity.py", "test_security_secrets.py"},
    "memory_search": {"test_security_secrets.py"},
    "memory_remove": {"test_security_integrity.py"},
    "spawn_agent": {"test_security_permissions.py"},
    "await_agent": {"test_security_permissions.py"},
    "collect_artifacts": {"test_security_integrity.py"},
    "bash": {"test_security_permissions.py"},
    "powershell": {"test_security_permissions.py"},
    "ask_user": {"test_security_permissions.py"},
}


def _base_raw(tmp_path: Path) -> dict:
    return {
        "global": {"global_folder": str(tmp_path / "global")},
        "ingest": {
            "roots": [str(tmp_path / "docs")],
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "llms": {
            "providers": [{"provider_name": "openai", "api_key": "test-key"}],
            "services": {"default": {"provider": "openai", "model": "gpt-5.2"}},
        },
        "vectordb": {
            "provider": "chromadb",
            "persist_dir": str(tmp_path / ".chroma"),
            "collection": "local_kb",
        },
        "query": {"mode": "grounded"},
        "remote_providers": [{"name": "arxiv", "type": "arxiv"}],
        "agents": {
            "enabled": True,
            "agents_folder": str(tmp_path / "Agents"),
            "max_tokens_budget": 8000,
            "max_iterations": 3,
            "context_window_strategy": "compact",
            "sandbox": {
                "allowed_read_paths": [str(tmp_path)],
                "allowed_write_paths": [str(tmp_path)],
                "allow_url_access": False,
                "require_user_permission": {},
                "tool_llm_assignments": {},
            },
            "agent_configs": {},
        },
    }


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in sorted(_TOOL_NAMES):
        registry.register(_StubTool(name))
    return registry


def _expected_tools(agent_cls, registry: ToolRegistry, *, allow_url_access: bool) -> set[str]:
    allowed = set(getattr(agent_cls, "tool_allowlist", set()) or set())
    allowed |= {"ask_user"}
    registered = {tool.name for tool in registry.list_tools()}
    allowed &= registered
    if not allow_url_access:
        builtin_query_tools = {"query_knowledge_base", "query_llm", "query_remote_chain"}
        network_tools = set(ToolSandbox._NETWORK_TOOL_NAMES)
        network_tools |= {
            name
            for name in allowed
            if name.startswith("query_") and name not in builtin_query_tools
        }
        allowed -= network_tools
    return allowed


@pytest.mark.parametrize("agent_cls", _AGENT_CLASSES)
def test_agent_tool_exposure_respects_allowlist_and_sandbox(
    tmp_path: Path,
    agent_cls,
) -> None:
    config = build_config_from_raw(_base_raw(tmp_path), tmp_path / "config.json")
    agents = config.agents
    assert agents is not None
    registry = _build_registry()
    sandbox = ToolSandbox(agents.sandbox)
    agent = agent_cls(
        config.llm,
        registry,
        sandbox,
        agents,
        lsm_config=config,
    )

    exposed = {item.get("name") for item in agent._get_tool_definitions(registry)}
    expected = _expected_tools(agent_cls, registry, allow_url_access=False)

    assert exposed == expected


def test_agent_tool_stride_coverage_matrix() -> None:
    covered = set()
    for agent_cls in _AGENT_CLASSES:
        allowed = set(getattr(agent_cls, "tool_allowlist", set()) or set())
        allowed |= {"ask_user"}
        covered |= allowed

    missing = sorted(name for name in covered if name not in _STRIDE_TEST_MODULES)
    assert not missing, f"Missing STRIDE coverage mapping for tools: {missing}"

    tests_root = Path("tests/test_agents")
    for tool_name, modules in _STRIDE_TEST_MODULES.items():
        for module in modules:
            assert (tests_root / module).exists(), (
                f"Expected security test module {module} for tool '{tool_name}'"
            )
