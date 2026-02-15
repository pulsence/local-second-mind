"""
Logging configuration for Local Second Mind.

Provides structured logging with configurable levels and formatting.
"""

import logging
import sys
from types import TracebackType
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for colored terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for different log levels."""

    # Define colors for each log level
    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD + Colors.BG_RED + Colors.WHITE,
    }

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True):
        """
        Initialize colored formatter.

        Args:
            fmt: Log message format
            datefmt: Date format
            use_colors: Whether to use ANSI colors (disable for file output)
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        if self.use_colors and record.levelno in self.LEVEL_COLORS:
            # Add color to level name
            level_color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"

        return super().format(record)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for LSM.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output

    Example:
        >>> setup_logging(level="DEBUG")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create root logger
    root_logger = logging.getLogger("lsm")
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Format: [LEVEL] message (timestamp for DEBUG)
    if numeric_level <= logging.DEBUG:
        console_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    else:
        console_format = "[%(levelname)s] %(message)s"
        console_formatter = ColoredFormatter(console_format)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)

        # More detailed format for file logs
        file_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
        file_formatter = ColoredFormatter(file_format, use_colors=False)

        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing file: %s", filename)
    """
    # Ensure all LSM loggers are under the 'lsm' namespace
    if not name.startswith("lsm"):
        name = f"lsm.{name}"

    return logging.getLogger(name)


def format_exception_summary(
    error: BaseException,
    *,
    max_length: int = 180,
) -> str:
    """
    Build a concise one-line exception summary for user-facing error messages.

    Args:
        error: Exception instance.
        max_length: Maximum output length.

    Returns:
        Single-line summary (trimmed when needed).
    """
    exception_name = error.__class__.__name__
    detail = str(error or "").strip()
    summary = exception_name if not detail else f"{exception_name}: {detail}"
    if max_length > 3 and len(summary) > max_length:
        return summary[: max_length - 3].rstrip() + "..."
    return summary


def exception_exc_info(
    error: BaseException,
) -> tuple[type[BaseException], BaseException, TracebackType | None]:
    """
    Build an ``exc_info`` tuple suitable for logger calls.

    Args:
        error: Exception instance.

    Returns:
        Tuple consumable by ``logging.Logger`` methods.
    """
    return (type(error), error, error.__traceback__)


# Convenience function for quick setup
def configure_logging_from_args(verbose: bool = False, log_level: Optional[str] = None,
                                log_file: Optional[str] = None) -> None:
    """
    Configure logging based on CLI arguments.

    Args:
        verbose: If True, set level to DEBUG
        log_level: Explicit log level (overrides verbose)
        log_file: Optional file for log output

    Example:
        >>> configure_logging_from_args(verbose=True)
        >>> configure_logging_from_args(log_level="WARNING")
    """
    if log_level:
        level = log_level.upper()
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"

    setup_logging(level=level, log_file=log_file)
