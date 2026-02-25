"""
Assistant agents and registry specs.
"""

from .assistant import AssistantAgent
from .email_assistant import EmailAssistantAgent

AGENT_SPECS = [
    {
        "name": "assistant",
        "agent_cls": AssistantAgent,
        "theme": "Assistants",
        "category": "Assistant",
    },
    {
        "name": "email_assistant",
        "agent_cls": EmailAssistantAgent,
        "theme": "Assistants",
        "category": "Email",
    },
]

__all__ = ["AssistantAgent", "EmailAssistantAgent", "AGENT_SPECS"]
