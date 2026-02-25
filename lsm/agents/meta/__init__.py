"""
Meta agents and task-graph helpers.
"""

from .meta import AssistantMetaAgent, MetaAgent
from .task_graph import AgentTask, TaskGraph, ParallelGroup

AGENT_SPECS = [
    {
        "name": "meta",
        "agent_cls": MetaAgent,
        "theme": "Meta",
        "category": "Orchestration",
    },
    {
        "name": "assistant_meta",
        "agent_cls": AssistantMetaAgent,
        "theme": "Meta",
        "category": "Orchestration",
    },
]

__all__ = [
    "MetaAgent",
    "AssistantMetaAgent",
    "AgentTask",
    "TaskGraph",
    "ParallelGroup",
    "AGENT_SPECS",
]
