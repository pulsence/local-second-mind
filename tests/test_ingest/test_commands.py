from lsm.ui.tui.commands.ingest import parse_tag_args


def test_parse_tag_args_max() -> None:
    max_chunks, error = parse_tag_args("--max 10")
    assert max_chunks == 10
    assert error is None
