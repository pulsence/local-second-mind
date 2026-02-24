"""
Productivity agents and registry specs.
"""

from .general import GeneralAgent
from .librarian import LibrarianAgent
from .manuscript_editor import ManuscriptEditorAgent
from .writing import WritingAgent

AGENT_SPECS = [
    {
        "name": "general",
        "agent_cls": GeneralAgent,
        "theme": "Productivity",
        "category": "General",
    },
    {
        "name": "librarian",
        "agent_cls": LibrarianAgent,
        "theme": "Productivity",
        "category": "Librarian",
    },
    {
        "name": "manuscript_editor",
        "agent_cls": ManuscriptEditorAgent,
        "theme": "Productivity",
        "category": "Manuscript Editor",
    },
    {
        "name": "writing",
        "agent_cls": WritingAgent,
        "theme": "Productivity",
        "category": "Writing",
    },
]

__all__ = [
    "GeneralAgent",
    "LibrarianAgent",
    "ManuscriptEditorAgent",
    "WritingAgent",
    "AGENT_SPECS",
]
