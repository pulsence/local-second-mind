"""
Stress tests for very large collections.

These tests are skipped by default. Set LSM_PERF_TEST=1 to run.
"""

import os
import pytest

from lsm.ingest.stats import get_collection_stats


class FakeCollection:
    """Minimal collection stub for stress testing."""

    def __init__(self, total_chunks: int, file_count: int) -> None:
        self._total = total_chunks
        self._file_count = file_count

    def count(self) -> int:
        return self._total

    def get(self, include=None, limit=None, offset=0, where=None):
        if limit is None:
            limit = self._total
        end = min(offset + limit, self._total)
        metadatas = []
        for i in range(offset, end):
            doc_index = i % self._file_count
            ext = ".pdf" if i % 3 == 0 else ".md"
            metadatas.append(
                {
                    "source_path": f"/stress/doc_{doc_index}{ext}",
                    "ext": ext,
                    "ingested_at": "2026-01-01T00:00:00Z",
                }
            )
        return {"metadatas": metadatas}


@pytest.mark.performance
@pytest.mark.skipif(not os.getenv("LSM_PERF_TEST"), reason="Set LSM_PERF_TEST=1 to run")
def test_stress_stats_full_scan():
    total_chunks = int(os.getenv("LSM_STRESS_CHUNKS", "1000000"))
    file_count = int(os.getenv("LSM_STRESS_FILES", "20000"))

    collection = FakeCollection(total_chunks=total_chunks, file_count=file_count)
    stats = get_collection_stats(collection, limit=None, batch_size=10000)

    assert stats["total_chunks"] == total_chunks
    assert stats["unique_files"] == file_count
