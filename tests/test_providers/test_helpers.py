"""
Tests for shared LLM provider helper utilities.
"""

from lsm.providers.helpers import (
    generate_fallback_answer,
    parse_json_payload,
    parse_ranking_response,
    prepare_candidates_for_rerank,
    strip_code_fences,
)


class TestProviderHelpers:
    def test_prepare_candidates_truncates_long_text(self):
        candidates = [
            {
                "text": "x" * 1300,
                "metadata": {"source_path": "/tmp/source.md", "chunk_index": 2},
                "distance": 0.25,
            }
        ]
        items = prepare_candidates_for_rerank(candidates, max_text_length=1200)

        assert len(items) == 1
        assert items[0]["index"] == 0
        assert items[0]["source_path"] == "/tmp/source.md"
        assert items[0]["text"].endswith("\n...[truncated]...")
        assert len(items[0]["text"]) <= 1200

    def test_parse_ranking_response_handles_invalid_indices(self):
        candidates = [
            {"text": "first"},
            {"text": "second"},
            {"text": "third"},
        ]
        ranking = [
            {"index": 50, "reason": "bad"},
            {"index": 1, "reason": "good"},
            {"index": "2", "reason": "also good"},
            {"index": 1, "reason": "duplicate"},
        ]

        result = parse_ranking_response(ranking, candidates, k=3)

        assert [item["text"] for item in result] == ["second", "third", "first"]

    def test_strip_code_fences_removes_markdown(self):
        raw = "```json\n{\"tags\": [\"one\"]}\n```"
        assert strip_code_fences(raw) == "{\"tags\": [\"one\"]}"

    def test_parse_json_payload_handles_fenced_json(self):
        raw = "```json\n{\"tags\": [\"alpha\", \"beta\"]}\n```"
        payload = parse_json_payload(raw)
        assert payload == {"tags": ["alpha", "beta"]}

    def test_fallback_answer_format(self):
        answer = generate_fallback_answer(
            question="What is X?",
            context="[S1] X is a concept.",
            provider_name="Provider X",
        )
        assert "Offline mode" in answer
        assert "Provider X unavailable" in answer
        assert "Question: What is X?" in answer
        assert "Retrieved context" in answer
