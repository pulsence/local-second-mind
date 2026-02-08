from __future__ import annotations

import time

from lsm.query.cache import QueryCache


def test_query_cache_set_get_and_build_key_stability() -> None:
    cache = QueryCache(ttl_seconds=60, max_size=5)
    key1 = cache.build_key("q", "grounded", {"a": 1}, 12, 6)
    key2 = cache.build_key("q", "grounded", {"a": 1}, 12, 6)
    assert key1 == key2

    cache.set(key1, {"answer": "ok"})
    assert cache.get(key1) == {"answer": "ok"}


def test_query_cache_ttl_expiration() -> None:
    cache = QueryCache(ttl_seconds=1, max_size=5)
    key = cache.build_key("q", "grounded", {}, 12, 6)
    cache.set(key, 123)
    assert cache.get(key) == 123
    time.sleep(1.1)
    assert cache.get(key) is None


def test_query_cache_lru_eviction() -> None:
    cache = QueryCache(ttl_seconds=60, max_size=2)
    k1 = cache.build_key("q1", "grounded", {}, 12, 6)
    k2 = cache.build_key("q2", "grounded", {}, 12, 6)
    k3 = cache.build_key("q3", "grounded", {}, 12, 6)
    cache.set(k1, 1)
    cache.set(k2, 2)
    assert cache.get(k1) == 1  # promote k1
    cache.set(k3, 3)
    assert cache.get(k2) is None
    assert cache.get(k1) == 1
    assert cache.get(k3) == 3
