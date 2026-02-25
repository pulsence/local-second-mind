"""
OAI-PMH (Open Archives Initiative Protocol for Metadata Harvesting) provider.

Provides a generic harvester for OAI-PMH compliant repositories, enabling
access to institutional repositories, digital libraries, and archives
worldwide.

Protocol specification: https://www.openarchives.org/pmh/
Version: OAI-PMH 2.0

Common repositories supporting OAI-PMH:
- arXiv (https://export.arxiv.org/oai2)
- PubMed Central (https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi)
- Zenodo (https://zenodo.org/oai2d)
- Figshare (https://api.figshare.com/v2/oai)
- SSRN (https://papers.ssrn.com/sol3/oai.cfm)
- RePEc (https://ideas.repec.org/cgi-bin/oai.pl)
- HAL (https://api.archives-ouvertes.fr/oai/hal)
- DOAJ (https://doaj.org/oai)
- Many institutional repositories (DSpace, EPrints, Fedora, etc.)
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Callable, Generator

from lsm.remote.base import RemoteResult
from lsm.logging import get_logger
from lsm.remote.providers.base_oai import (
    BaseOAIProvider,
    DataciteParser,
    DublinCoreParser,
    MARCParser,
    METADATA_PARSERS,
    MetadataParser,
    OAIPMHClient,
    OAIPMHError,
    OAIRecord,
    OAIRepository,
)

logger = get_logger(__name__)


# ============================================================================
# Repository Configuration
# ============================================================================

# Pre-configured well-known repositories
KNOWN_REPOSITORIES: Dict[str, OAIRepository] = {
    "arxiv": OAIRepository(
        name="arXiv",
        base_url="https://export.arxiv.org/oai2",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:arXiv.org:",
        url_template="https://arxiv.org/abs/{id}",
        description="Open access archive for physics, mathematics, computer science, and more",
        min_interval_seconds=3.0,  # arXiv requires 3s between requests
    ),
    "pubmed": OAIRepository(
        name="PubMed Central",
        base_url="https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:pubmedcentral.nih.gov:",
        url_template="https://www.ncbi.nlm.nih.gov/pmc/articles/{id}",
        description="Biomedical and life sciences journal literature",
        min_interval_seconds=1.0,
    ),
    "zenodo": OAIRepository(
        name="Zenodo",
        base_url="https://zenodo.org/oai2d",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:zenodo.org:",
        url_template="https://zenodo.org/record/{id}",
        description="General-purpose open repository for research outputs",
        min_interval_seconds=1.0,
    ),
    "doaj": OAIRepository(
        name="Directory of Open Access Journals",
        base_url="https://doaj.org/oai",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:doaj.org/article:",
        url_template="https://doaj.org/article/{id}",
        description="Open access journals across all disciplines",
        min_interval_seconds=1.0,
    ),
    "hal": OAIRepository(
        name="HAL (Hyper Articles en Ligne)",
        base_url="https://api.archives-ouvertes.fr/oai/hal",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:HAL:",
        url_template="https://hal.archives-ouvertes.fr/{id}",
        description="French national open archive for scholarly documents",
        min_interval_seconds=1.0,
    ),
    "repec": OAIRepository(
        name="RePEc (Research Papers in Economics)",
        base_url="https://ideas.repec.org/cgi-bin/oai.pl",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:RePEc:",
        url_template="https://ideas.repec.org/{id}",
        description="Economics and related sciences working papers and articles",
        min_interval_seconds=2.0,
    ),
    "europeana": OAIRepository(
        name="Europeana",
        base_url="https://oai.europeana.eu/oai",
        metadata_prefix="oai_dc",
        identifier_prefix="",
        url_template=None,
        description="European cultural heritage digital collections",
        min_interval_seconds=1.0,
    ),
    "philpapers": OAIRepository(
        name="PhilPapers",
        base_url="https://philpapers.org/oai.pl",
        metadata_prefix="oai_dc",
        identifier_prefix="oai:philpapers.org:",
        url_template="https://philpapers.org/rec/{id}",
        description="Philosophy research bibliography",
        min_interval_seconds=2.0,
    ),
    "dspace_demo": OAIRepository(
        name="DSpace Demo",
        base_url="https://demo.dspace.org/server/oai/request",
        metadata_prefix="oai_dc",
        identifier_prefix="",
        url_template=None,
        description="DSpace demonstration repository",
        min_interval_seconds=1.0,
    ),
}


# ============================================================================
# OAI-PMH Remote Provider
# ============================================================================

class OAIPMHProvider(BaseOAIProvider):
    """
    Remote provider for OAI-PMH compliant repositories.

    This provider enables harvesting metadata from institutional repositories,
    digital libraries, and archives that implement the OAI-PMH protocol.

    Supports:
    - Pre-configured well-known repositories (arXiv, Zenodo, PubMed, etc.)
    - Custom repository endpoints
    - Multiple metadata formats (Dublin Core, MARC, DataCite)
    - Resumption tokens for large result sets
    - Set-based filtering
    """

    DEFAULT_TIMEOUT = 30
    DEFAULT_MIN_INTERVAL_SECONDS = 1.0
    DEFAULT_SNIPPET_MAX_CHARS = 700
    DEFAULT_METADATA_PREFIX = "oai_dc"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OAI-PMH provider.

        Args:
            config: Configuration dict with keys:
                - repository: Name of known repository OR custom base URL
                - repositories: List of repository names/URLs to search (multi-repo mode)
                - metadata_prefix: Preferred metadata format (default: oai_dc)
                - set_spec: Optional set to filter by
                - timeout: Request timeout in seconds (default: 30)
                - min_interval_seconds: Rate limiting (default: 1.0)
                - snippet_max_chars: Max snippet length (default: 700)
                - user_agent: Custom User-Agent header
                - custom_repositories: Dict of custom repository configs
        """
        super().__init__(config)

        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        self.metadata_prefix = config.get("metadata_prefix") or self.DEFAULT_METADATA_PREFIX
        self.set_spec = config.get("set_spec")
        self.user_agent = (
            config.get("user_agent")
            or os.getenv("LSM_OAI_PMH_USER_AGENT")
            or "LocalSecondMind/1.0 (OAI-PMH Harvester)"
        )

        # Build repository list
        self.repositories: List[OAIRepository] = []
        self._load_repositories(config)

        # Create clients for each repository
        self._clients: Dict[str, OAIPMHClient] = {}
        for repo in self.repositories:
            self._clients[repo.name] = OAIPMHClient(
                base_url=repo.base_url,
                timeout=self.timeout,
                user_agent=self.user_agent,
                min_interval_seconds=repo.min_interval_seconds,
            )

    def _load_repositories(self, config: Dict[str, Any]) -> None:
        """Load repository configurations."""
        # Load custom repositories first
        custom_repos = config.get("custom_repositories") or {}
        for name, repo_config in custom_repos.items():
            self.repositories.append(OAIRepository(
                name=repo_config.get("name", name),
                base_url=repo_config["base_url"],
                metadata_prefix=repo_config.get("metadata_prefix", self.metadata_prefix),
                set_spec=repo_config.get("set_spec", self.set_spec),
                identifier_prefix=repo_config.get("identifier_prefix", ""),
                url_template=repo_config.get("url_template"),
                description=repo_config.get("description", ""),
                min_interval_seconds=repo_config.get(
                    "min_interval_seconds", self.min_interval_seconds
                ),
            ))

        # Single repository mode
        if "repository" in config:
            repo_id = config["repository"]
            if repo_id in KNOWN_REPOSITORIES:
                self.repositories.append(KNOWN_REPOSITORIES[repo_id])
            elif repo_id.startswith("http"):
                # Treat as custom URL
                self.repositories.append(OAIRepository(
                    name="Custom Repository",
                    base_url=repo_id,
                    metadata_prefix=self.metadata_prefix,
                    set_spec=self.set_spec,
                ))

        # Multi-repository mode
        if "repositories" in config:
            for repo_id in config["repositories"]:
                if repo_id in KNOWN_REPOSITORIES:
                    if not any(r.name == KNOWN_REPOSITORIES[repo_id].name for r in self.repositories):
                        self.repositories.append(KNOWN_REPOSITORIES[repo_id])
                elif repo_id.startswith("http"):
                    if not any(r.base_url == repo_id for r in self.repositories):
                        self.repositories.append(OAIRepository(
                            name=f"Repository ({repo_id[:30]}...)",
                            base_url=repo_id,
                            metadata_prefix=self.metadata_prefix,
                        ))

        # Default to Zenodo if no repositories specified
        if not self.repositories:
            self.repositories.append(KNOWN_REPOSITORIES["zenodo"])

    @property
    def name(self) -> str:
        return "oai_pmh"

    def is_available(self) -> bool:
        """OAI-PMH repositories are generally freely available."""
        return len(self.repositories) > 0

    def validate_config(self) -> None:
        """Validate OAI-PMH configuration."""
        if not self.repositories:
            raise ValueError("At least one OAI-PMH repository must be configured.")

        for repo in self.repositories:
            if not repo.base_url:
                raise ValueError(f"Repository '{repo.name}' is missing base_url.")

    def get_name(self) -> str:
        """Get provider name."""
        if len(self.repositories) == 1:
            return f"OAI-PMH ({self.repositories[0].name})"
        return f"OAI-PMH ({len(self.repositories)} repositories)"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Text query used for client-side filtering.", "required": True},
            {"name": "repository", "type": "string", "description": "Repository name or base URL hint.", "required": False},
            {"name": "set_spec", "type": "string", "description": "OAI-PMH set filter.", "required": False},
            {"name": "metadata_prefix", "type": "string", "description": "Requested metadata format.", "required": False},
            {"name": "keywords", "type": "array[string]", "description": "Additional keywords.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "oai_identifier", "type": "string", "description": "OAI-PMH identifier."},
            {"name": "repository", "type": "string", "description": "Repository display name."},
            {"name": "subjects", "type": "array[string]", "description": "Record subjects."},
            {"name": "types", "type": "array[string]", "description": "Resource types."},
            {"name": "publisher", "type": "string", "description": "Publisher name."},
            {"name": "source", "type": "string", "description": "Source collection/journal."},
            {"name": "languages", "type": "array[string]", "description": "Language tags."},
            {"name": "datestamp", "type": "string", "description": "OAI datestamp."},
            {"name": "sets", "type": "array[string]", "description": "OAI set spec values."},
            {"name": "citation", "type": "string", "description": "Formatted citation string."},
        ]

    def get_description(self) -> str:
        return "OAI-PMH harvester for repository metadata with client-side relevance filtering."

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        return super().search_structured(input_dict, max_results=max_results)

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        """
        Search OAI-PMH repositories.

        Note: OAI-PMH doesn't support full-text search directly. This method
        fetches recent records and filters them client-side. For better search,
        use repository-specific APIs when available.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of RemoteResult objects
        """
        if not query.strip():
            return []

        logger.debug(f"OAI-PMH: query='{query}', max_results={max_results}")

        all_results: List[RemoteResult] = []

        # Search each repository
        results_per_repo = max(3, max_results // len(self.repositories))

        for repo in self.repositories:
            try:
                repo_results = self._search_repository(repo, query, results_per_repo)
                all_results.extend(repo_results)
            except Exception as exc:
                logger.warning(f"OAI-PMH error for {repo.name}: {exc}")
                continue

        # Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        results = all_results[:max_results]

        # Re-score for final ranking
        for i, result in enumerate(results):
            result.score = max(0.2, 1.0 - (i * 0.8 / max(1, max_results - 1)))

        logger.info(f"OAI-PMH returned {len(results)} results")
        return results

    def _search_repository(
        self, repo: OAIRepository, query: str, max_results: int
    ) -> List[RemoteResult]:
        """Search a single repository."""
        client = self._clients[repo.name]

        # Fetch records (OAI-PMH doesn't support search, so we list and filter)
        records_to_fetch = max_results * 5  # Fetch more to filter

        try:
            raw_records, _ = client.list_records(
                metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
                set_spec=repo.set_spec or self.set_spec,
            )
        except OAIPMHError as exc:
            logger.warning(f"OAI-PMH error for {repo.name}: {exc}")
            return []

        # Parse and filter records
        results: List[RemoteResult] = []
        query_terms = query.lower().split()

        for header, metadata_elem in raw_records[:records_to_fetch]:
            if header.get("deleted"):
                continue

            if metadata_elem is None:
                continue

            # Parse metadata
            record = self._parse_record(header, metadata_elem, repo)
            if record is None:
                continue

            # Filter by query terms
            searchable_text = f"{record.title} {record.description} {' '.join(record.subjects)}".lower()
            if not any(term in searchable_text for term in query_terms):
                continue

            # Convert to RemoteResult
            result = self._record_to_result(record, repo, len(results), max_results)
            if result:
                results.append(result)

            if len(results) >= max_results:
                break

        return results

    # ========================================================================
    # Additional Methods for Advanced Usage
    # ========================================================================

    def list_available_repositories(self) -> List[Dict[str, str]]:
        """
        List all known OAI-PMH repositories.

        Returns:
            List of dicts with name, base_url, description
        """
        repos = []
        for key, repo in KNOWN_REPOSITORIES.items():
            repos.append({
                "key": key,
                "name": repo.name,
                "base_url": repo.base_url,
                "description": repo.description,
            })
        return repos

    def identify_repository(self, repository: str) -> Dict[str, Any]:
        """
        Get identification information for a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            Dict with repository identification info
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return client.identify()

    def list_repository_sets(self, repository: str) -> List[Dict[str, str]]:
        """
        List available sets/collections in a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            List of set dicts
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return list(client.list_sets())

    def list_metadata_formats(self, repository: str) -> List[Dict[str, str]]:
        """
        List available metadata formats for a repository.

        Args:
            repository: Repository name key or base URL

        Returns:
            List of metadata format dicts
        """
        if repository in KNOWN_REPOSITORIES:
            base_url = KNOWN_REPOSITORIES[repository].base_url
        else:
            base_url = repository

        client = OAIPMHClient(
            base_url=base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
        )
        return client.list_metadata_formats()

    def get_record(
        self, identifier: str, repository: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific record by OAI identifier.

        Args:
            identifier: Full OAI identifier
            repository: Repository to query (if not in identifier)

        Returns:
            Parsed record dict or None
        """
        # Determine repository from identifier or parameter
        if repository:
            if repository in KNOWN_REPOSITORIES:
                repo = KNOWN_REPOSITORIES[repository]
            else:
                repo = OAIRepository(name="Custom", base_url=repository)
        elif self.repositories:
            repo = self.repositories[0]
        else:
            raise ValueError("No repository specified")

        client = self._clients.get(repo.name)
        if client is None:
            client = OAIPMHClient(
                base_url=repo.base_url,
                timeout=self.timeout,
                user_agent=self.user_agent,
            )

        result = client.get_record(
            identifier=identifier,
            metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
        )

        if result is None:
            return None

        header, metadata_elem = result
        record = self._parse_record(header, metadata_elem, repo)

        if record is None:
            return None

        return {
            "identifier": record.identifier,
            "title": record.title,
            "creators": record.creators,
            "description": record.description,
            "subjects": record.subjects,
            "date": record.date,
            "year": record.year,
            "publisher": record.publisher,
            "source": record.source,
            "identifiers": record.identifiers,
            "types": record.types,
            "languages": record.languages,
            "datestamp": record.datestamp,
            "sets": record.set_specs,
        }

    def harvest_records(
        self,
        repository: str,
        max_records: int = 100,
        from_date: Optional[str] = None,
        until_date: Optional[str] = None,
        set_spec: Optional[str] = None,
        callback: Optional[Callable[[OAIRecord], None]] = None,
    ) -> List[OAIRecord]:
        """
        Harvest records from a repository.

        Args:
            repository: Repository name key or base URL
            max_records: Maximum records to harvest
            from_date: Start date (YYYY-MM-DD)
            until_date: End date (YYYY-MM-DD)
            set_spec: Optional set to filter by
            callback: Optional callback for each record

        Returns:
            List of OAIRecord objects
        """
        if repository in KNOWN_REPOSITORIES:
            repo = KNOWN_REPOSITORIES[repository]
        else:
            repo = OAIRepository(name="Custom", base_url=repository)

        client = OAIPMHClient(
            base_url=repo.base_url,
            timeout=self.timeout,
            user_agent=self.user_agent,
            min_interval_seconds=repo.min_interval_seconds,
        )

        records: List[OAIRecord] = []
        resumption_token = None

        while len(records) < max_records:
            try:
                raw_records, next_token = client.list_records(
                    metadata_prefix=repo.metadata_prefix or self.metadata_prefix,
                    from_date=from_date,
                    until_date=until_date,
                    set_spec=set_spec or repo.set_spec,
                    resumption_token=resumption_token,
                )
            except OAIPMHError as exc:
                if exc.code == "noRecordsMatch":
                    break
                raise

            for header, metadata_elem in raw_records:
                if len(records) >= max_records:
                    break

                if header.get("deleted"):
                    continue

                if metadata_elem is None:
                    continue

                record = self._parse_record(header, metadata_elem, repo)
                if record:
                    records.append(record)
                    if callback:
                        callback(record)

            if next_token:
                resumption_token = next_token
            else:
                break

        return records
