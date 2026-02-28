"""Tests for generic provider helper utilities."""

from lsm.providers.helpers import UnsupportedParamTracker, parse_json_payload


class TestProviderHelpers:
    def test_parse_json_payload_handles_fenced_json(self):
        raw = "```json\n{\"tags\": [\"alpha\", \"beta\"]}\n```"
        payload = parse_json_payload(raw)
        assert payload == {"tags": ["alpha", "beta"]}

    def test_parse_json_payload_handles_inline_fragment(self):
        raw = "noise before {\"ok\": true} and after"
        payload = parse_json_payload(raw)
        assert payload == {"ok": True}

    def test_unsupported_param_tracker_roundtrip(self):
        tracker = UnsupportedParamTracker()
        assert tracker.should_send("model-x", "foo") is True
        tracker.mark_unsupported("model-x", "foo")
        assert tracker.should_send("model-x", "foo") is False
        assert tracker.should_send("model-x", "bar") is True
