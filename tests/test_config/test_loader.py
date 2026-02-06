from lsm.config.loader import parse_config_text


def test_parse_config_text_json() -> None:
    parsed = parse_config_text('{"a": 1}', "config.json")
    assert parsed["a"] == 1

