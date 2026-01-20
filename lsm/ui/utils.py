"""
UI utility helpers.
"""

from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path
from typing import Dict, Any, List

from lsm.config.models import LSMConfig
from lsm.logging import get_logger
from lsm.remote import create_remote_provider
from lsm.query.session import SessionState

logger = get_logger(__name__)


def display_provider_name(name: str) -> str:
    """
    Normalize provider name for display.

    Args:
        name: Raw provider name

    Returns:
        Display-friendly provider name
    """
    if name in {"anthropic", "claude"}:
        return "claude"
    return name


def format_feature_label(feature: str) -> str:
    """
    Format a feature name for display.

    Args:
        feature: Raw feature name

    Returns:
        Short label for display
    """
    return {
        "query": "query",
        "tagging": "tag",
        "ranking": "rerank",
    }.get(feature, feature)


def open_file(path: str) -> bool:
    """
    Open file with system default application.

    Cross-platform: Windows, macOS, Linux.

    Args:
        path: Path to file to open

    Returns:
        True if the open command was issued successfully, False otherwise.
    """
    if not path or not os.path.exists(path):
        return False

    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return True
    except Exception as exc:
        logger.error(f"Failed to open file: {exc}")
        return False


def run_remote_search(
    provider_name: str,
    query: str,
    config: LSMConfig,
    max_results: int = 5,
) -> str:
    """
    Run a test search on a specific remote provider.

    Args:
        provider_name: Name of the provider to search
        query: Search query
        config: LSM configuration
        max_results: Maximum results to return

    Returns:
        Formatted search results string
    """
    import time

    provider_config = None
    for pc in config.remote_providers or []:
        if pc.name.lower() == provider_name.lower():
            provider_config = pc
            break

    if not provider_config:
        return f"Provider not found: {provider_name}\nUse /remote-providers to see available providers.\n"

    lines = [
        "",
        f"Searching '{provider_name}' for: {query}",
        "-" * 60,
    ]

    try:
        start_time = time.time()

        provider = create_remote_provider(
            provider_config.type,
            {
                "type": provider_config.type,
                "enabled": provider_config.enabled,
                "weight": provider_config.weight,
                "api_key": provider_config.api_key,
                "endpoint": provider_config.endpoint,
                "max_results": provider_config.max_results,
                "language": provider_config.language,
                "user_agent": provider_config.user_agent,
                "timeout": provider_config.timeout,
                "min_interval_seconds": provider_config.min_interval_seconds,
                "section_limit": provider_config.section_limit,
                "snippet_max_chars": provider_config.snippet_max_chars,
                "include_disambiguation": provider_config.include_disambiguation,
            }
        )

        results = provider.search(query, max_results=max_results)
        elapsed = time.time() - start_time

        if not results:
            lines.append(f"No results found. ({elapsed:.2f}s)")
            lines.append("")
            return "\n".join(lines)

        lines.append(f"Found {len(results)} results in {elapsed:.2f}s:\n")

        for i, result in enumerate(results, 1):
            title = result.title or "(no title)"
            url = result.url or ""
            snippet = result.snippet or ""
            score = result.score

            lines.append(f"{i}. {title}")
            if url:
                lines.append(f"   URL: {url}")
            if score is not None:
                lines.append(f"   Score: {score:.3f}")
            if snippet:
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"   {snippet}")
            lines.append("")

    except Exception as exc:
        logger.error(f"Remote search failed: {exc}")
        lines.append(f"Search failed: {exc}\n")

    return "\n".join(lines)


def run_remote_search_all(
    query: str,
    config: LSMConfig,
    state: SessionState,
) -> str:
    """
    Run a search across all enabled remote providers.

    Args:
        query: Search query
        config: LSM configuration
        state: Session state to store results

    Returns:
        Formatted search results string
    """
    import time

    active_providers = config.get_active_remote_providers()

    if not active_providers:
        return "No enabled remote providers configured.\nUse /remote-providers to see available providers.\n"

    lines = [
        "",
        f"Searching {len(active_providers)} providers for: {query}",
        "=" * 60,
    ]

    all_results: list[Dict[str, Any]] = []
    start_time = time.time()

    for provider_config in active_providers:
        provider_name = provider_config.name
        lines.append(f"\n[{provider_name}]")

        try:
            provider = create_remote_provider(
                provider_config.type,
                {
                    "type": provider_config.type,
                    "enabled": provider_config.enabled,
                    "weight": provider_config.weight,
                    "api_key": provider_config.api_key,
                    "endpoint": provider_config.endpoint,
                    "max_results": provider_config.max_results,
                    "language": provider_config.language,
                    "user_agent": provider_config.user_agent,
                    "timeout": provider_config.timeout,
                    "min_interval_seconds": provider_config.min_interval_seconds,
                    "section_limit": provider_config.section_limit,
                    "snippet_max_chars": provider_config.snippet_max_chars,
                    "include_disambiguation": provider_config.include_disambiguation,
                }
            )

            max_results = provider_config.max_results or 5
            results = provider.search(query, max_results=max_results)

            if not results:
                lines.append("  No results")
                continue

            lines.append(f"  Found {len(results)} results")

            for result in results:
                all_results.append({
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "score": result.score,
                    "provider": provider_name,
                    "weight": provider_config.weight,
                    "weighted_score": (result.score or 0.5) * provider_config.weight,
                    "metadata": result.metadata,
                })

        except Exception as exc:
            logger.error(f"Search failed for {provider_name}: {exc}")
            lines.append(f"  Error: {exc}")

    elapsed = time.time() - start_time

    all_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)

    seen_urls = set()
    unique_results: list[Dict[str, Any]] = []
    for result in all_results:
        url = result.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique_results.append(result)

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"Total: {len(unique_results)} unique results in {elapsed:.2f}s")
    lines.append("=" * 60)

    for i, result in enumerate(unique_results[:10], 1):
        title = result.get("title") or "(no title)"
        provider = result.get("provider", "")
        weighted_score = result.get("weighted_score", 0)
        url = result.get("url", "")

        lines.append(f"\n{i}. [{provider}] {title}")
        lines.append(f"   Weighted Score: {weighted_score:.3f}")
        if url:
            lines.append(f"   URL: {url}")

    lines.append("")

    state.last_remote_sources = unique_results

    return "\n".join(lines)


def format_source_list(sources: List[Dict[str, Any]]) -> str:
    """
    Format sources list for display.

    Groups sources by file and shows citation labels.

    Args:
        sources: List of source metadata dicts

    Returns:
        Formatted sources string

    Example:
        >>> sources = [
        ...     {"label": "S1", "source_path": "/docs/readme.md", "source_name": "readme.md"},
        ...     {"label": "S2", "source_path": "/docs/readme.md", "source_name": "readme.md"},
        ...     {"label": "S3", "source_path": "/docs/guide.md", "source_name": "guide.md"},
        ... ]
        >>> print(format_source_list(sources))
        Sources:
        - [S1] [S2] readme.md — /docs/readme.md
        - [S3] guide.md — /docs/guide.md
    """
    if not sources:
        return ""

    logger.debug(f"Formatting {len(sources)} sources for display")

    lines = ["", "Sources:"]
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    # Group sources by path
    for s in sources:
        path = (s.get("source_path") or "unknown").strip()
        label = (s.get("label") or "").strip()
        name = (s.get("source_name") or Path(path).name or "unknown").strip()

        if path not in grouped:
            grouped[path] = {
                "name": name,
                "labels": [],
            }
            order.append(path)

        if label and label not in grouped[path]["labels"]:
            grouped[path]["labels"].append(label)

    # Format grouped sources
    for path in order:
        entry = grouped[path]
        labels = " ".join(f"[{lbl}]" for lbl in entry["labels"])
        lines.append(f"- {labels} {entry['name']} — {path}")

    return "\n".join(lines)
