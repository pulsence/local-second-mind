from __future__ import annotations

from lsm.agents import (
    AssistantAgent,
    GeneralAgent,
    LibrarianAgent,
    ManuscriptEditorAgent,
    ResearchAgent,
    WritingAgent,
)
from lsm.agents.factory import AgentRegistry


def test_agent_registry_metadata_and_grouping() -> None:
    registry = AgentRegistry()
    research_entry = registry.get_entry("research")
    assert research_entry is not None
    assert research_entry.theme == "Academic"
    assert research_entry.category == "Research"
    assert research_entry.description == ResearchAgent.description
    assert research_entry.risk_posture == "network"

    groups = registry.list_groups()
    themes = [group.theme for group in groups]
    assert "Academic" in themes
    academic_group = next(group for group in groups if group.theme == "Academic")
    academic_names = [entry.name for entry in academic_group.entries]
    assert "curator" in academic_names
    assert "research" in academic_names
    assert "synthesis" in academic_names


def test_agent_registry_exposes_core_catalog_metadata() -> None:
    registry = AgentRegistry()
    general_entry = registry.get_entry("general")
    assert general_entry is not None
    assert general_entry.description == GeneralAgent.description
    assert general_entry.tool_allowlist is not None
    assert "read_file" in general_entry.tool_allowlist
    assert general_entry.risk_posture == "writes_workspace"

    librarian_entry = registry.get_entry("librarian")
    assert librarian_entry is not None
    assert librarian_entry.description == LibrarianAgent.description
    assert librarian_entry.tool_allowlist is not None
    assert "query_knowledge_base" in librarian_entry.tool_allowlist

    assistant_entry = registry.get_entry("assistant")
    assert assistant_entry is not None
    assert assistant_entry.description == AssistantAgent.description
    assert assistant_entry.risk_posture == "writes_workspace"

    manuscript_entry = registry.get_entry("manuscript_editor")
    assert manuscript_entry is not None
    assert manuscript_entry.description == ManuscriptEditorAgent.description


def test_agent_reexports_preserve_top_level_imports() -> None:
    assert ResearchAgent.__module__ == "lsm.agents.academic.research"
    assert WritingAgent.__module__ == "lsm.agents.productivity.writing"
