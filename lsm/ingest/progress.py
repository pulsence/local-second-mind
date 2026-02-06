from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Progress:
    total_files: int = 0
    seen: int = 0
    updated: int = 0
    skipped: int = 0
    empty: int = 0
    errors: int = 0

    chunks: int = 0
    writes: int = 0

    t0: float = field(default_factory=time.time)
    last_report: float = field(default_factory=time.time)
    report_every_s: float = 1.0
    on_update: Optional[Callable[[str, int, int, str], None]] = None

    def _emit(self, event: str, message: str = "") -> None:
        if self.on_update:
            self.on_update(event, self.seen, self.total_files, message)

    def start(self, total_files: int) -> None:
        self.total_files = total_files
        self.t0 = time.time()
        self.last_report = self.t0
        self._emit("start", f"Found {total_files} files to consider.")

    def file_done(self) -> None:
        self.seen += 1
        self.maybe_report()

    def file_skipped(self) -> None:
        self.skipped += 1
        self.file_done()

    def file_empty(self) -> None:
        self.empty += 1
        self.file_done()

    def file_updated(self, chunk_count: int) -> None:
        self.updated += 1
        self.chunks += chunk_count
        self.file_done()

    def write_done(self, n_chunks: int) -> None:
        self.writes += 1

    def error(self, msg: str) -> None:
        self.errors += 1
        self._emit("error", msg)

    def maybe_report(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_report) < self.report_every_s:
            return
        self.last_report = now

        elapsed = now - self.t0
        rate = (self.seen / elapsed) if elapsed > 0 else 0.0

        line = (
            f"Progress: {self.seen}/{self.total_files} files | "
            f"updated={self.updated} skipped={self.skipped} empty={self.empty} errors={self.errors} | "
            f"chunks={self.chunks} writes={self.writes} | "
            f"{rate:.1f} files/s | {elapsed:.1f}s elapsed"
        )
        self._emit("progress", line)

    def finish(self) -> None:
        self.maybe_report(force=True)
        self._emit("finish", "Ingest complete.")
