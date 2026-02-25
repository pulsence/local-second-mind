"""
Integration coverage for one provider per remote category.
"""

from unittest.mock import Mock, patch

import pytest

from lsm.remote.factory import create_remote_provider
from lsm.remote.validation import validate_output


def _mock_json_response(payload: dict) -> Mock:
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload
    response.text = ""
    response.raise_for_status = lambda: None
    return response


def test_factory_validates_required_api_key() -> None:
    with pytest.raises(ValueError, match="Guardian requires an API key"):
        create_remote_provider("guardian", {})


@patch("requests.get")
def test_academic_provider_integration_crossref(mock_get: Mock) -> None:
    mock_get.return_value = _mock_json_response(
        {
            "message": {
                "items": [
                    {
                        "title": ["Example Paper"],
                        "DOI": "10.1234/example",
                        "URL": "https://example.com/paper",
                        "author": [{"given": "Ada", "family": "Lovelace"}],
                        "published": {"date-parts": [[2021, 1, 1]]},
                        "is-referenced-by-count": 5,
                        "container-title": ["Journal of Examples"],
                        "publisher": "Example Publisher",
                        "type": "journal-article",
                    }
                ]
            }
        }
    )

    provider = create_remote_provider("crossref", {})
    results = provider.search("example", max_results=1)

    assert len(results) == 1
    assert validate_output(results) == []


@patch("requests.get")
def test_cultural_provider_integration_archive_org(mock_get: Mock) -> None:
    mock_get.return_value = _mock_json_response(
        {
            "response": {
                "docs": [
                    {
                        "identifier": "item1",
                        "title": "Archive Item",
                        "description": "Archive description",
                        "creator": "Author",
                        "year": "1999",
                        "mediatype": "texts",
                        "collection": ["collection1"],
                        "subject": ["history"],
                        "downloads": 42,
                    }
                ]
            }
        }
    )

    provider = create_remote_provider("archive_org", {})
    results = provider.search("archive", max_results=1)

    assert len(results) == 1
    assert validate_output(results) == []


@patch("requests.get")
def test_news_provider_integration_guardian(mock_get: Mock) -> None:
    mock_get.return_value = _mock_json_response(
        {
            "response": {
                "results": [
                    {
                        "id": "guardian/1",
                        "webTitle": "Guardian Title",
                        "webUrl": "https://example.com/news",
                        "webPublicationDate": "2024-01-01T00:00:00Z",
                        "sectionName": "World",
                        "fields": {"trailText": "Summary"},
                    }
                ]
            }
        }
    )

    provider = create_remote_provider("guardian", {"api_key": "test"})
    results = provider.search("world", max_results=1)

    assert len(results) == 1
    assert validate_output(results) == []


@patch("requests.get")
def test_web_provider_integration_brave(mock_get: Mock) -> None:
    mock_get.return_value = _mock_json_response(
        {
            "web": {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                        "description": "Snippet",
                        "age": "1d",
                        "language": "en",
                        "family_friendly": True,
                    }
                ]
            }
        }
    )

    provider = create_remote_provider("web_search", {"api_key": "test"})
    results = provider.search("example", max_results=1)

    assert len(results) == 1
    assert validate_output(results) == []
