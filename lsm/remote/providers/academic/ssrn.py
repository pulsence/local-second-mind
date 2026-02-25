"""
SSRN provider using OAI-PMH harvesting.
"""

from __future__ import annotations

from typing import Any, Dict

from lsm.remote.providers.academic.oai_pmh import OAIPMHProvider


class SSRNProvider(OAIPMHProvider):
    """
    SSRN preprints via OAI-PMH.
    """

    def __init__(self, config: Dict[str, Any]):
        merged = dict(config)
        merged.setdefault("repository", "ssrn")
        super().__init__(merged)

    @property
    def name(self) -> str:
        return "ssrn"

    def get_name(self) -> str:
        return "SSRN"

    def get_description(self) -> str:
        return "SSRN working papers and preprints via OAI-PMH."
