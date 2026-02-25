"""
Project MUSE provider using OAI-PMH harvesting.
"""

from __future__ import annotations

from typing import Any, Dict

from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider


class ProjectMUSEProvider(OAIPMHProvider):
    """
    Project MUSE humanities metadata via OAI-PMH.
    """

    def __init__(self, config: Dict[str, Any]):
        merged = dict(config)
        merged.setdefault("repository", "project_muse")
        super().__init__(merged)

    @property
    def name(self) -> str:
        return "project_muse"

    def get_name(self) -> str:
        return "Project MUSE"

    def get_description(self) -> str:
        return "Project MUSE humanities and social sciences via OAI-PMH."
