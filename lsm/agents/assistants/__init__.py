"""
Assistant agents and registry specs.
"""

from .assistant import AssistantAgent
from .email_assistant import EmailAssistantAgent
from .calendar_assistant import CalendarAssistantAgent
from .news_assistant import NewsAssistantAgent

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
    {
        "name": "calendar_assistant",
        "agent_cls": CalendarAssistantAgent,
        "theme": "Assistants",
        "category": "Calendar",
    },
    {
        "name": "news_assistant",
        "agent_cls": NewsAssistantAgent,
        "theme": "Assistants",
        "category": "News",
    },
]

__all__ = [
    "AssistantAgent",
    "EmailAssistantAgent",
    "CalendarAssistantAgent",
    "NewsAssistantAgent",
    "AGENT_SPECS",
]
