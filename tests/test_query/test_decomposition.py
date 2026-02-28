from __future__ import annotations

from lsm.query.decomposition import decompose_query, extract_fields_ai, extract_fields_deterministic


def test_extract_fields_deterministic_doi_author_years() -> None:
    fields = extract_fields_deterministic(
        'Find papers by John Smith between 2021 and 2024 DOI 10.1000/xyz123 title: "Vector Search"'
    )
    assert fields.author is not None
    assert fields.doi == "10.1000/xyz123"
    assert fields.date_range == ("2021", "2024")
    assert fields.title is not None
    assert "vector" in fields.keywords


def test_decompose_query_ai_method_falls_back_cleanly() -> None:
    fields = decompose_query("papers about embeddings", method="ai", llm_config=object())
    assert fields.raw_query == "papers about embeddings"
    assert "embeddings" in fields.keywords


def test_extract_fields_ai_uses_provider_json_response(monkeypatch) -> None:
    class _MockProvider:
        def send_message(self, input, instruction=None, **kwargs):
            _ = input, instruction, kwargs
            return (
                '{"author":"Jane Doe","keywords":["retrieval","embeddings"],'
                '"title":"Dense Retrieval","date_range":{"start":"2020","end":"2024"},'
                '"doi":"10.1234/abcd","raw_query":"custom"}'
            )

    monkeypatch.setattr("lsm.query.decomposition.create_provider", lambda cfg: _MockProvider())

    fields = extract_fields_ai("find papers", llm_config=object())
    assert fields.author == "Jane Doe"
    assert fields.title == "Dense Retrieval"
    assert fields.doi == "10.1234/abcd"
    assert fields.date_range == ("2020", "2024")
    assert "retrieval" in fields.keywords
    assert fields.raw_query == "custom"


def test_extract_fields_ai_falls_back_on_invalid_json(monkeypatch) -> None:
    class _MockProvider:
        def send_message(self, input, instruction=None, **kwargs):
            _ = input, instruction, kwargs
            return "not-json"

    monkeypatch.setattr("lsm.query.decomposition.create_provider", lambda cfg: _MockProvider())
    fields = extract_fields_ai("papers by John Smith in 2024", llm_config=object())

    assert fields.author is not None
    assert fields.date_range == ("2024", "2024")
