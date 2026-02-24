"""
Academic agents and registry specs.
"""

from .curator import CuratorAgent
from .research import ResearchAgent
from .synthesis import SynthesisAgent

AGENT_SPECS = [
    {
        "name": "curator",
        "agent_cls": CuratorAgent,
        "theme": "Academic",
        "category": "Curation",
    },
    {
        "name": "research",
        "agent_cls": ResearchAgent,
        "theme": "Academic",
        "category": "Research",
    },
    {
        "name": "synthesis",
        "agent_cls": SynthesisAgent,
        "theme": "Academic",
        "category": "Synthesis",
    },
]

__all__ = [
    "CuratorAgent",
    "ResearchAgent",
    "SynthesisAgent",
    "AGENT_SPECS",
]
