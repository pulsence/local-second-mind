"""
RSS/Atom feed provider with caching support.
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional

import requests

from lsm.logging import get_logger
from lsm.remote.base import BaseRemoteProvider, RemoteResult
from lsm.remote.storage import load_feed_cache, save_feed_cache

logger = get_logger(__name__)

_RSS_NAMESPACES = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"


@dataclass
class FeedItem:
    identifier: str
    title: str
    link: str
    summary: str
    published: str
    authors: List[str]
    categories: List[str]


class RSSProvider(BaseRemoteProvider):
    """
    Remote provider for RSS 2.0 and Atom feeds.
    """

    DEFAULT_TIMEOUT = 15
    DEFAULT_MIN_INTERVAL_SECONDS = 0.0
    DEFAULT_SNIPPET_MAX_CHARS = 700
    DEFAULT_CACHE_TTL = 1800
    DEFAULT_USER_AGENT = "LocalSecondMind/1.0 (rss)"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RSS provider.

        Args:
            config: Configuration dict with optional keys:
                - feeds / feed_urls / feed_url: feed URL list or single URL
                - timeout: Request timeout in seconds (default: 15)
                - min_interval_seconds: Minimum seconds between requests
                - snippet_max_chars: Max characters per snippet (default: 700)
                - cache_ttl_seconds / cache_ttl: Cache TTL in seconds (default: 1800)
                - user_agent: Custom User-Agent header
                - global_folder: Override global folder for cache storage
                - fetcher: Optional callable for fetching feed content (testing)
        """
        super().__init__(config)

        self.feed_urls = _normalize_feed_urls(config)
        self.timeout = config.get("timeout") or self.DEFAULT_TIMEOUT
        min_interval = config.get("min_interval_seconds")
        self.min_interval_seconds = float(
            min_interval if min_interval is not None else self.DEFAULT_MIN_INTERVAL_SECONDS
        )
        snippet_max_chars = config.get("snippet_max_chars")
        self.snippet_max_chars = int(
            snippet_max_chars if snippet_max_chars is not None else self.DEFAULT_SNIPPET_MAX_CHARS
        )
        cache_ttl = config.get("cache_ttl_seconds", config.get("cache_ttl"))
        self.cache_ttl_seconds = int(cache_ttl if cache_ttl is not None else self.DEFAULT_CACHE_TTL)
        self.user_agent = (
            config.get("user_agent")
            or os.getenv("LSM_RSS_USER_AGENT")
            or self.DEFAULT_USER_AGENT
        )
        self.global_folder = config.get("global_folder")
        self.vectordb_path = config.get("vectordb_path")
        fetcher = config.get("fetcher")
        self._fetcher = fetcher if callable(fetcher) else None
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "rss"

    def validate_config(self) -> None:
        if self.cache_ttl_seconds < 1:
            raise ValueError("cache_ttl_seconds must be positive")

    def search(self, query: str, max_results: int = 5) -> List[RemoteResult]:
        if not self.feed_urls:
            return []
        return self._search_feed_urls(self.feed_urls, query or "", max_results=max_results)

    def search_structured(
        self,
        input_dict: Dict[str, Any],
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        feed_urls = _extract_feed_urls(input_dict)
        if not feed_urls:
            feed_urls = list(self.feed_urls)
        query = self._compose_query_from_input(input_dict)
        results = self._search_feed_urls(feed_urls, query, max_results=max_results)
        return [self._to_structured_output(result) for result in results]

    def get_name(self) -> str:
        return "RSS"

    def get_input_fields(self) -> List[Dict[str, Any]]:
        return [
            {"name": "query", "type": "string", "description": "Query filter.", "required": False},
            {"name": "feed_url", "type": "string", "description": "Single feed URL.", "required": False},
            {"name": "feeds", "type": "array[string]", "description": "List of feed URLs.", "required": False},
        ]

    def get_output_fields(self) -> List[Dict[str, Any]]:
        return super().get_output_fields() + [
            {"name": "published", "type": "string", "description": "Published timestamp."},
            {"name": "authors", "type": "array[string]", "description": "Feed item authors."},
            {"name": "categories", "type": "array[string]", "description": "Feed categories/tags."},
            {"name": "feed_url", "type": "string", "description": "Originating feed URL."},
            {"name": "item_id", "type": "string", "description": "Feed item identifier."},
            {"name": "year", "type": "integer", "description": "Publication year when available."},
        ]

    def get_description(self) -> str:
        return "RSS/Atom feed provider with caching and seen-item tracking."

    def _search_feed_urls(
        self,
        feed_urls: Iterable[str],
        query: str,
        *,
        max_results: int,
    ) -> List[RemoteResult]:
        results: List[RemoteResult] = []
        normalized_query = str(query or "").strip().lower()

        for feed_url in feed_urls:
            if len(results) >= max_results:
                break
            cache = load_feed_cache(
                feed_url=feed_url,
                global_folder=self.global_folder,
                max_age=self.cache_ttl_seconds,
                vectordb_path=self.vectordb_path,
            )
            seen_ids = set(cache.seen_ids) if cache else set()

            fetched = False
            if cache and cache.fresh:
                items = [FeedItem(**item) for item in _normalize_cached_items(cache.items)]
            else:
                items = self._fetch_feed_items(feed_url)
                fetched = True

            new_items = []
            for item in items:
                if item.identifier and item.identifier not in seen_ids:
                    new_items.append(item)
            seen_ids.update([item.identifier for item in items if item.identifier])

            if fetched or cache is None:
                try:
                    save_feed_cache(
                        feed_url=feed_url,
                        items=[item.__dict__ for item in items],
                        seen_ids=list(seen_ids),
                        global_folder=self.global_folder,
                        vectordb_path=self.vectordb_path,
                        cache_ttl_seconds=self.cache_ttl_seconds,
                    )
                except Exception as exc:
                    logger.warning("Failed saving feed cache for '%s': %s", feed_url, exc)

            for item in new_items:
                if normalized_query and not _matches_query(item, normalized_query):
                    continue
                results.append(self._to_remote_result(feed_url, item))
                if len(results) >= max_results:
                    break

        return results

    def _fetch_feed_items(self, feed_url: str) -> List[FeedItem]:
        self._throttle()
        try:
            content = self._fetch_feed_content(feed_url)
        except requests.exceptions.RequestException as exc:
            logger.error("RSS fetch error (%s): %s", feed_url, exc)
            return []
        except Exception as exc:
            logger.error("RSS fetch failure (%s): %s", feed_url, exc)
            return []

        try:
            return _parse_feed_items(content)
        except Exception as exc:
            logger.error("RSS parse error (%s): %s", feed_url, exc)
            return []

    def _fetch_feed_content(self, feed_url: str) -> str:
        if self._fetcher is not None:
            return str(self._fetcher(feed_url))
        headers = {"User-Agent": self.user_agent}
        response = requests.get(feed_url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    def _throttle(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def _to_remote_result(self, feed_url: str, item: FeedItem) -> RemoteResult:
        snippet = _truncate_snippet(item.summary, self.snippet_max_chars)
        year = _extract_year(item.published)
        metadata = {
            "feed_url": feed_url,
            "published": item.published,
            "authors": item.authors,
            "categories": item.categories,
            "item_id": item.identifier,
        }
        if year is not None:
            metadata["year"] = year
        return RemoteResult(
            title=item.title or "Untitled",
            url=item.link or feed_url,
            snippet=snippet,
            score=1.0,
            metadata=metadata,
        )


def _normalize_feed_urls(config: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    urls.extend(_extract_feed_urls(config))
    return _dedupe_urls(urls)


def _extract_feed_urls(input_dict: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    if not isinstance(input_dict, dict):
        return urls
    if input_dict.get("feed_url"):
        urls.append(str(input_dict.get("feed_url")).strip())
    if input_dict.get("feed"):
        urls.append(str(input_dict.get("feed")).strip())
    feeds = input_dict.get("feeds") or input_dict.get("feed_urls")
    if isinstance(feeds, list):
        urls.extend([str(url).strip() for url in feeds if str(url).strip()])
    elif feeds:
        urls.append(str(feeds).strip())
    return _dedupe_urls(urls)


def _dedupe_urls(urls: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    cleaned: List[str] = []
    for url in urls:
        value = str(url).strip()
        if not value or value in seen:
            continue
        cleaned.append(value)
        seen.add(value)
    return cleaned


def _parse_feed_items(content: str) -> List[FeedItem]:
    root = ET.fromstring(content)
    tag = _strip_ns(root.tag)
    if tag == "feed":
        return _parse_atom_entries(root)
    if tag == "rss" or root.find("channel") is not None:
        return _parse_rss_items(root)
    return []


def _parse_rss_items(root: ET.Element) -> List[FeedItem]:
    channel = root.find("channel") if _strip_ns(root.tag) == "rss" else root
    if channel is None:
        return []
    items: List[FeedItem] = []
    for item in channel.findall("item"):
        title = _first_text(item, ["title"])
        link = _first_text(item, ["link"])
        guid = _first_text(item, ["guid"])
        summary = _first_text(item, ["content:encoded", "description"])
        published = _first_text(item, ["pubDate", "dc:date"])
        author = _first_text(item, ["author", "dc:creator"])
        authors = _split_authors(author)
        categories = [
            text for text in (elem.text for elem in item.findall("category")) if text
        ]
        identifier = guid or link or _fallback_id(title, published)
        items.append(
            FeedItem(
                identifier=identifier,
                title=title,
                link=link,
                summary=summary,
                published=published,
                authors=authors,
                categories=categories,
            )
        )
    return items


def _parse_atom_entries(root: ET.Element) -> List[FeedItem]:
    ns = {"atom": _ATOM_NAMESPACE}
    items: List[FeedItem] = []
    for entry in root.findall("atom:entry", ns):
        title = _atom_text(entry, "title", ns)
        summary = _atom_text(entry, "summary", ns) or _atom_text(entry, "content", ns)
        link = _atom_link(entry, ns)
        identifier = _atom_text(entry, "id", ns) or link or _fallback_id(title, "")
        published = _atom_text(entry, "published", ns) or _atom_text(entry, "updated", ns)
        authors = []
        for author in entry.findall("atom:author", ns):
            name = _atom_text(author, "name", ns)
            if name:
                authors.append(name)
        categories = []
        for category in entry.findall("atom:category", ns):
            term = category.get("term") or category.get("label")
            if term:
                categories.append(term)
        items.append(
            FeedItem(
                identifier=identifier,
                title=title,
                link=link,
                summary=summary,
                published=published,
                authors=authors,
                categories=categories,
            )
        )
    return items


def _normalize_cached_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "identifier": str(item.get("identifier", "")).strip(),
                "title": str(item.get("title", "")).strip(),
                "link": str(item.get("link", "")).strip(),
                "summary": str(item.get("summary", "")).strip(),
                "published": str(item.get("published", "")).strip(),
                "authors": list(item.get("authors") or []),
                "categories": list(item.get("categories") or []),
            }
        )
    return normalized


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _first_text(element: ET.Element, tags: List[str]) -> str:
    for tag in tags:
        if ":" in tag:
            prefix, local = tag.split(":", 1)
            namespace = _RSS_NAMESPACES.get(prefix)
            if namespace:
                value = element.findtext(f".//{{{namespace}}}{local}")
            else:
                value = None
        else:
            value = element.findtext(tag)
        if value:
            return value.strip()
    return ""


def _atom_text(element: ET.Element, tag: str, ns: Dict[str, str]) -> str:
    value = element.findtext(f"atom:{tag}", default="", namespaces=ns)
    return value.strip()


def _atom_link(element: ET.Element, ns: Dict[str, str]) -> str:
    links = element.findall("atom:link", ns)
    for link in links:
        rel = (link.get("rel") or "alternate").lower()
        href = link.get("href") or ""
        if rel == "alternate" and href:
            return href.strip()
    if links:
        href = links[0].get("href") or ""
        return href.strip()
    return ""


def _split_authors(author_text: str) -> List[str]:
    value = str(author_text or "").strip()
    if not value:
        return []
    if "," in value:
        parts = value.split(",")
    elif " and " in value:
        parts = value.split(" and ")
    else:
        parts = [value]
    return [part.strip() for part in parts if part.strip()]


def _fallback_id(title: str, published: str) -> str:
    base = f"{title}|{published}".strip()
    return base if base else "item"


def _matches_query(item: FeedItem, query: str) -> bool:
    haystack = f"{item.title} {item.summary}".lower()
    return query in haystack


def _truncate_snippet(text: str, limit: int) -> str:
    cleaned = re.sub(r"\\s+", " ", str(text or "")).strip()
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _extract_year(published: str) -> Optional[int]:
    value = str(published or "").strip()
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        return dt.year
    except Exception:
        try:
            return parsedate_to_datetime(value).year
        except Exception:
            return None
