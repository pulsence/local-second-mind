"""
Plain-text logging utilities with configurable verbosity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, TextIO
import sys


class LogVerbosity(IntEnum):
    """
    Verbosity levels for plain-text logs.
    """

    NORMAL = 0
    VERBOSE = 1
    DEBUG = 2


def normalize_verbosity(
    value: str | int | LogVerbosity | None,
    *,
    default: LogVerbosity = LogVerbosity.NORMAL,
) -> LogVerbosity:
    """
    Normalize a verbosity value into a LogVerbosity enum.
    """
    if isinstance(value, LogVerbosity):
        return value
    if value is None:
        return default
    if isinstance(value, int):
        try:
            return LogVerbosity(value)
        except ValueError:
            return default
    normalized = str(value).strip().lower()
    mapping = {
        "normal": LogVerbosity.NORMAL,
        "verbose": LogVerbosity.VERBOSE,
        "debug": LogVerbosity.DEBUG,
    }
    return mapping.get(normalized, default)


@dataclass
class PlainTextLogger:
    """
    Lightweight logger that writes plain-text messages to a stream and/or file.
    """

    verbosity: LogVerbosity = LogVerbosity.NORMAL
    stream: Optional[TextIO] = sys.stdout
    file_path: Optional[Path] = None

    def log(
        self,
        message: str,
        *,
        level: str | int | LogVerbosity = LogVerbosity.NORMAL,
        end: str = "\n",
    ) -> None:
        """
        Write a message when its level is within the logger verbosity.
        """
        message_level = normalize_verbosity(level, default=LogVerbosity.NORMAL)
        if message_level > self.verbosity:
            return
        self.write(f"{message}{end}")

    def write(self, text: str, *, append: bool = True) -> None:
        """
        Write raw text to configured outputs.
        """
        if self.stream is not None:
            try:
                self.stream.write(text)
                self.stream.flush()
            except Exception:
                pass
        if self.file_path is None:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with self.file_path.open(mode, encoding="utf-8") as handle:
            handle.write(text)


def create_plaintext_logger(
    *,
    stream: Optional[TextIO] = sys.stdout,
    file_path: Optional[Path] = None,
    verbosity: str | int | LogVerbosity = LogVerbosity.NORMAL,
) -> PlainTextLogger:
    """
    Create a PlainTextLogger with the requested outputs and verbosity.
    """
    return PlainTextLogger(
        verbosity=normalize_verbosity(verbosity, default=LogVerbosity.NORMAL),
        stream=stream,
        file_path=file_path,
    )
