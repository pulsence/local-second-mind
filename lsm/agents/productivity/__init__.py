"""
Productivity agents and registry specs.
"""

from .writing import WritingAgent

AGENT_SPECS = [
    {
        "name": "writing",
        "agent_cls": WritingAgent,
        "theme": "Productivity",
        "category": "Writing",
    },
]

__all__ = ["WritingAgent", "AGENT_SPECS"]
