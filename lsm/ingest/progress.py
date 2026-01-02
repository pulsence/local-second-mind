from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional

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

    def start(self, total_files: int) -> None:
        self.total_files = total_files
        self.t0 = time.time()
        self.last_report = self.t0
        self._print(f"Found {total_files} files to consider.")

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
        # You may want to count chunks written separately; we track chunks at file_updated time.

    def error(self, msg: str) -> None:
        self.errors += 1
        self._print(f"[ERROR] {msg}")

    def maybe_report(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_report) < self.report_every_s:
            return
        self.last_report = now

        elapsed = now - self.t0
        rate = (self.seen / elapsed) if elapsed > 0 else 0.0

        # single-line “status bar” style update
        line = (
            f"Progress: {self.seen}/{self.total_files} files | "
            f"updated={self.updated} skipped={self.skipped} empty={self.empty} errors={self.errors} | "
            f"chunks={self.chunks} writes={self.writes} | "
            f"{rate:.1f} files/s | {elapsed:.1f}s elapsed"
        )
        self._print(line, overwrite=True)

    def finish(self) -> None:
        self.maybe_report(force=True)
        self._print("\nIngest complete.")

    def _print(self, s: str, overwrite: bool = False) -> None:
        if overwrite:
            # carriage return + clear-to-eol (works in most terminals)
            sys.stdout.write("\r" + s + " " * max(0, 4))
            sys.stdout.flush()
        else:
            sys.stdout.write(s + "\n")
            sys.stdout.flush()
