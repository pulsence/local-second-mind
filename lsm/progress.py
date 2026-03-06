from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


def format_eta(eta_seconds: float) -> str:
    total_seconds = max(0, int(eta_seconds))
    if total_seconds < 60:
        return f"ETA {total_seconds}s"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"ETA {minutes}m {seconds:02d}s"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"ETA {hours}h {minutes:02d}m"


@dataclass
class MovingAverageETA:
    max_samples: int = 8
    min_samples: int = 2
    _samples: Deque[tuple[float, int]] = field(default_factory=deque, init=False)

    def reset(self) -> None:
        self._samples.clear()

    def add_sample(self, progress: int, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else float(timestamp)
        value = int(progress)

        if self._samples and value < self._samples[-1][1]:
            self._samples.clear()
        if self._samples and value == self._samples[-1][1]:
            return

        self._samples.append((now, value))
        while len(self._samples) > max(1, int(self.max_samples)):
            self._samples.popleft()

    def format(self, current: int, total: int) -> str:
        if int(current) >= int(total):
            return "done"
        rate = self.rate()
        if rate is None or rate <= 0:
            return "estimating..."
        remaining = max(0, int(total) - int(current))
        return format_eta(remaining / rate)

    def rate(self) -> float | None:
        if len(self._samples) < max(2, int(self.min_samples)):
            return None

        start_time, start_progress = self._samples[0]
        end_time, end_progress = self._samples[-1]
        elapsed = end_time - start_time
        delta = end_progress - start_progress
        if elapsed <= 0 or delta <= 0:
            return None
        return delta / elapsed
