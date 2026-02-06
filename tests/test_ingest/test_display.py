from lsm.ingest.display import get_help


def test_get_help_contains_ingest_commands_header() -> None:
    assert "INGEST COMMANDS" in get_help()

