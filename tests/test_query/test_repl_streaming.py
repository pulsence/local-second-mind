"""
Tests for REPL streaming helpers.
"""

from lsm.query.display import stream_output


def test_stream_output_combines_chunks(capsys):
    result = stream_output(["Hello", " ", "world"])
    captured = capsys.readouterr()
    assert "Typing..." in captured.out
    assert "Hello world" in captured.out
    assert result == "Hello world"
