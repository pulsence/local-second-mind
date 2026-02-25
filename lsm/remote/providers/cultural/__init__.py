"""Cultural heritage and archive providers."""

from __future__ import annotations

__all__ = [
    "ArchiveOrgProvider",
    "DPLAProvider",
    "LOCProvider",
    "SmithsonianProvider",
    "MetProvider",
    "RijksmuseumProvider",
    "IIIFProvider",
    "WikidataProvider",
]

from lsm.remote.providers.cultural.archive_org import ArchiveOrgProvider
from lsm.remote.providers.cultural.dpla import DPLAProvider
from lsm.remote.providers.cultural.loc import LOCProvider
from lsm.remote.providers.cultural.smithsonian import SmithsonianProvider
from lsm.remote.providers.cultural.met import MetProvider
from lsm.remote.providers.cultural.rijksmuseum import RijksmuseumProvider
from lsm.remote.providers.cultural.iiif import IIIFProvider
from lsm.remote.providers.cultural.wikidata import WikidataProvider
