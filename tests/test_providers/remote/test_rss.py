from __future__ import annotations

from pathlib import Path

import pytest

from lsm.remote.providers.rss import RSSProvider


RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Example Feed</title>
    <item>
      <title>First Item</title>
      <link>https://example.com/1</link>
      <guid>item-1</guid>
      <description>First summary</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
      <dc:creator>Jane Doe</dc:creator>
      <category>News</category>
    </item>
    <item>
      <title>Second Item</title>
      <link>https://example.com/2</link>
      <guid>item-2</guid>
      <description>Second summary</description>
      <pubDate>Tue, 02 Jan 2024 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Atom</title>
  <entry>
    <title>Atom One</title>
    <id>tag:example.com,2024:1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <summary>Atom summary</summary>
    <link rel="alternate" href="https://example.com/a1"/>
    <author>
      <name>John Smith</name>
    </author>
  </entry>
</feed>
"""


def test_rss_parses_rss2_feed(tmp_path: Path) -> None:
    provider = RSSProvider(
        {
            "feed_url": "https://example.com/rss",
            "fetcher": lambda _: RSS_FEED,
            "global_folder": tmp_path,
        }
    )
    results = provider.search("", max_results=5)
    assert len(results) == 2
    assert results[0].title == "First Item"
    assert results[0].url == "https://example.com/1"
    assert "First summary" in results[0].snippet
    assert results[0].metadata["authors"] == ["Jane Doe"]
    assert results[0].metadata["categories"] == ["News"]
    assert results[0].metadata["item_id"] == "item-1"
    assert results[0].metadata["year"] == 2024


def test_rss_parses_atom_feed(tmp_path: Path) -> None:
    provider = RSSProvider(
        {
            "feed_url": "https://example.com/atom",
            "fetcher": lambda _: ATOM_FEED,
            "global_folder": tmp_path,
        }
    )
    results = provider.search("", max_results=5)
    assert len(results) == 1
    assert results[0].title == "Atom One"
    assert results[0].url == "https://example.com/a1"
    assert results[0].metadata["authors"] == ["John Smith"]


def test_rss_cache_ttl_avoids_refetch(tmp_path: Path) -> None:
    calls = {"count": 0}

    def fetcher(_: str) -> str:
        calls["count"] += 1
        return RSS_FEED

    provider = RSSProvider(
        {
            "feed_url": "https://example.com/rss",
            "fetcher": fetcher,
            "cache_ttl_seconds": 3600,
            "global_folder": tmp_path,
        }
    )
    first = provider.search("", max_results=5)
    second = provider.search("", max_results=5)

    assert calls["count"] == 1
    assert len(first) == 2
    assert second == []


def test_rss_seen_item_tracking_returns_only_new_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    feeds = [
        RSS_FEED,
        RSS_FEED.replace("</channel>", """
    <item>
      <title>Third Item</title>
      <link>https://example.com/3</link>
      <guid>item-3</guid>
      <description>Third summary</description>
      <pubDate>Wed, 03 Jan 2024 00:00:00 GMT</pubDate>
    </item>
  </channel>
"""),
    ]
    calls = {"count": 0}

    def fetcher(_: str) -> str:
        value = feeds[calls["count"]]
        calls["count"] += 1
        return value

    provider = RSSProvider(
        {
            "feed_url": "https://example.com/rss",
            "fetcher": fetcher,
            "cache_ttl_seconds": 1,
            "global_folder": tmp_path,
        }
    )

    monkeypatch.setattr("lsm.remote.storage.time.time", lambda: 1000)
    first = provider.search("", max_results=5)

    monkeypatch.setattr("lsm.remote.storage.time.time", lambda: 2002)
    second = provider.search("", max_results=5)

    assert [item.title for item in first] == ["First Item", "Second Item"]
    assert [item.title for item in second] == ["Third Item"]
