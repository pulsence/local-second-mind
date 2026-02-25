"""
PhilArchive provider using OAI-PMH harvesting.
"""

from __future__ import annotations

from typing import Any, Dict

from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider


class PhilArchiveProvider(OAIPMHProvider):
    """
    PhilArchive philosophy preprints via OAI-PMH.
    """

    def __init__(self, config: Dict[str, Any]):
        merged = dict(config)
        merged.setdefault("repository", "philarchive")
        super().__init__(merged)

    @property
    def name(self) -> str:
        return "philarchive"

    def get_name(self) -> str:
        return "PhilArchive"

    def get_description(self) -> str:
        return "PhilArchive open-access philosophy papers via OAI-PMH."
