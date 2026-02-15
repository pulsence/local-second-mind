import logging

from lsm.logging import (
    ColoredFormatter,
    configure_logging_from_args,
    exception_exc_info,
    format_exception_summary,
    get_logger,
    setup_logging,
)


def test_get_logger_prefixes_lsm_namespace() -> None:
    logger = get_logger("module")
    assert logger.name == "lsm.module"


def test_colored_formatter_formats_message() -> None:
    formatter = ColoredFormatter("[%(levelname)s] %(message)s", use_colors=False)
    record = logging.LogRecord(
        name="lsm.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "[INFO] hello" in rendered


def test_setup_logging_with_file_handler(tmp_path) -> None:
    log_file = tmp_path / "lsm.log"
    setup_logging(level="DEBUG", log_file=str(log_file))
    logger = get_logger("test")
    logger.debug("debug entry")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "debug entry" in content


def test_configure_logging_from_args_selects_level(monkeypatch) -> None:
    calls = []

    def _fake_setup(level: str = "INFO", log_file=None):
        calls.append((level, log_file))

    monkeypatch.setattr("lsm.logging.setup_logging", _fake_setup)
    configure_logging_from_args(verbose=True, log_level=None, log_file=None)
    configure_logging_from_args(verbose=False, log_level="WARNING", log_file="x.log")

    assert calls[0][0] == "DEBUG"
    assert calls[1] == ("WARNING", "x.log")


def test_format_exception_summary_truncates_long_messages() -> None:
    error = RuntimeError("x" * 300)
    summary = format_exception_summary(error, max_length=40)
    assert summary.startswith("RuntimeError: ")
    assert summary.endswith("...")
    assert len(summary) == 40


def test_exception_exc_info_contains_traceback() -> None:
    captured: Exception | None = None
    try:
        raise ValueError("bad value")
    except ValueError as exc:
        captured = exc

    assert captured is not None
    exc_info = exception_exc_info(captured)
    assert exc_info[0] is ValueError
    assert exc_info[1] is captured
    assert exc_info[2] is captured.__traceback__
