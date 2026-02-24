"""
Assistant agents and registry specs.
"""

from .assistant import AssistantAgent

AGENT_SPECS = [
    {
        "name": "assistant",
        "agent_cls": AssistantAgent,
        "theme": "Assistants",
        "category": "Assistant",
    },
]

__all__ = ["AssistantAgent", "AGENT_SPECS"]
