from lsm.ui.utils import get_ingest_help


def test_get_help_contains_ingest_commands_header() -> None:
    assert "INGEST COMMANDS" in get_ingest_help()
