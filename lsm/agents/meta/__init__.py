"""
Meta agents and task-graph helpers.
"""

from .meta import MetaAgent
from .task_graph import AgentTask, TaskGraph

AGENT_SPECS = [
    {
        "name": "meta",
        "agent_cls": MetaAgent,
        "theme": "Meta",
        "category": "Orchestration",
    },
]

__all__ = ["MetaAgent", "AgentTask", "TaskGraph", "AGENT_SPECS"]
