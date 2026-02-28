"""
Tests for AI tagging module.

Tests cover:
- JSON serialization/deserialization for metadata storage
- Tag generation with retry logic
- Generic provider pattern usage
- OpenAI Responses API integration
- Untagged chunk retrieval with filtering
- User tag management
- Tag parsing fallback strategies
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

from lsm.vectordb.base import VectorDBGetResult
from lsm.ingest.tagging import (
    _serialize_tags,
    _deserialize_tags,
    generate_tags_for_chunk,
    get_tag_instructions,
    get_untagged_chunks,
    tag_chunks,
    add_user_tags,
    remove_user_tags,
    get_all_tags,
)
from lsm.config.models import LLMConfig


def _mock_provider(**overrides):
    """Create a mock BaseVectorDBProvider with sensible defaults."""
    provider = Mock()
    provider.get.return_value = overrides.get(
        "get_result", VectorDBGetResult()
    )
    return provider


class TestTagSerialization:
    """Test JSON serialization helpers for metadata compatibility."""

    def test_serialize_tags_basic(self):
        """Test basic tag list serialization to JSON string."""
        tags = ["python", "tutorial", "programming"]
        result = _serialize_tags(tags)

        assert isinstance(result, str)
        assert result == '["python", "tutorial", "programming"]'

    def test_serialize_empty_list(self):
        """Test serializing empty tag list."""
        result = _serialize_tags([])
        assert result == "[]"

    def test_deserialize_tags_json_string(self):
        """Test deserializing valid JSON string."""
        tags_json = '["python", "tutorial", "programming"]'
        result = _deserialize_tags(tags_json)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "python" in result

    def test_deserialize_tags_none(self):
        """Test deserializing None value."""
        result = _deserialize_tags(None)
        assert result == []

    def test_deserialize_tags_empty_string(self):
        """Test deserializing empty string."""
        result = _deserialize_tags("")
        assert result == []

    def test_deserialize_tags_invalid_json(self):
        """Test handling invalid JSON gracefully."""
        result = _deserialize_tags("{invalid json")
        assert result == []

    def test_deserialize_tags_non_list_json(self):
        """Test handling JSON that's not a list."""
        result = _deserialize_tags('{"tags": ["python"]}')
        assert result == []

    def test_round_trip_serialization(self):
        """Test serialize -> deserialize preserves data."""
        original = ["python", "machine-learning", "data-science"]
        serialized = _serialize_tags(original)
        deserialized = _deserialize_tags(serialized)

        assert deserialized == original


class TestTagGeneration:
    """Test tag generation with retry logic and provider abstraction."""

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_basic(self, mock_create_provider):
        """Test basic tag generation using generic provider."""
        # Mock provider
        mock_provider = Mock()
        mock_provider.send_message.return_value = '{"tags": ["python", "programming", "tutorial"]}'
        mock_create_provider.return_value = mock_provider

        # Create LLM config
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        # Generate tags
        tags = generate_tags_for_chunk(
            text="This is a Python programming tutorial",
            llm_config=llm_config,
            num_tags=3,
        )

        assert len(tags) == 3
        assert "python" in tags
        assert "programming" in tags
        assert "tutorial" in tags

        # Verify provider was created and called
        mock_create_provider.assert_called_once_with(llm_config)
        mock_provider.send_message.assert_called_once()

    def test_get_tag_instructions_includes_existing_context(self):
        instructions = get_tag_instructions(
            num_tags=3,
            existing_tags=["python", "testing"],
        )
        assert "Existing tags in this knowledge base: python, testing" in instructions
        assert "Include exactly 3 tags." in instructions

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_with_existing_context(self, mock_create_provider):
        """Test tag generation with existing tags context."""
        mock_provider = Mock()
        mock_provider.send_message.return_value = '{"tags": ["machine-learning", "python"]}'
        mock_create_provider.return_value = mock_provider

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        existing_tags = ["python", "data-science", "tutorial"]

        tags = generate_tags_for_chunk(
            text="Machine learning with Python",
            llm_config=llm_config,
            num_tags=2,
            existing_tags=existing_tags,
        )

        assert len(tags) <= 2

        # Verify existing_tags are included in instruction prompt
        call_kwargs = mock_provider.send_message.call_args.kwargs
        assert "Existing tags in this knowledge base" in call_kwargs["instruction"]
        assert "python, data-science, tutorial" in call_kwargs["instruction"]

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_retry_on_empty(self, mock_create_provider):
        """Test retry logic when LLM returns empty results."""
        mock_provider = Mock()
        # First call returns empty, second succeeds
        mock_provider.send_message.side_effect = [
            '{"tags": []}',  # First attempt fails
            '{"tags": ["python", "tutorial"]}',  # Second attempt succeeds
        ]
        mock_create_provider.return_value = mock_provider

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tags = generate_tags_for_chunk(
            text="Python tutorial",
            llm_config=llm_config,
            num_tags=3,
            max_retries=1,
        )

        # Should get tags from second attempt
        assert len(tags) == 2
        assert mock_provider.send_message.call_count == 2

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_retry_on_exception(self, mock_create_provider):
        """Test retry logic when LLM raises exception."""
        mock_provider = Mock()
        # First call raises, second succeeds
        mock_provider.send_message.side_effect = [
            Exception("API error"),
            '{"tags": ["python"]}',
        ]
        mock_create_provider.return_value = mock_provider

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tags = generate_tags_for_chunk(
            text="Python",
            llm_config=llm_config,
            max_retries=1,
        )

        assert len(tags) == 1
        assert mock_provider.send_message.call_count == 2

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_retry_exhausted(self, mock_create_provider):
        """Test that exception is raised after retries exhausted."""
        mock_provider = Mock()
        mock_provider.send_message.side_effect = Exception("Persistent API error")
        mock_create_provider.return_value = mock_provider

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        with pytest.raises(Exception, match="Persistent API error"):
            generate_tags_for_chunk(
                text="Test",
                llm_config=llm_config,
                max_retries=1,
            )

        # Should try twice (initial + 1 retry)
        assert mock_provider.send_message.call_count == 2

    @patch("lsm.ingest.tagging.create_provider")
    def test_generate_tags_empty_after_retries(self, mock_create_provider):
        """Test that empty list returned when all retries return empty."""
        mock_provider = Mock()
        mock_provider.send_message.return_value = '{"tags": []}'
        mock_create_provider.return_value = mock_provider

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tags = generate_tags_for_chunk(
            text="Test",
            llm_config=llm_config,
            max_retries=1,
        )

        assert tags == []
        assert mock_provider.send_message.call_count == 2


class TestUntaggedChunks:
    """Test finding untagged chunks with manual filtering."""

    def test_get_untagged_chunks_all_untagged(self):
        """Test getting untagged chunks when none have ai_tags."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1", "chunk2", "chunk3"],
                metadatas=[
                    {"source_path": "/docs/file1.md"},
                    {"source_path": "/docs/file2.md"},
                    {"source_path": "/docs/file3.md"},
                ],
                documents=["Text 1", "Text 2", "Text 3"],
            ),
        )

        untagged = get_untagged_chunks(provider, batch_size=10)

        assert len(untagged) == 3
        assert untagged[0]["id"] == "chunk1"
        assert untagged[0]["text"] == "Text 1"
        assert untagged[1]["id"] == "chunk2"
        assert untagged[2]["id"] == "chunk3"

    def test_get_untagged_chunks_filters_tagged(self):
        """Test that chunks with ai_tags are filtered out."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1", "chunk2", "chunk3"],
                metadatas=[
                    {"source_path": "/docs/file1.md"},  # No ai_tags
                    {"source_path": "/docs/file2.md", "ai_tags": '["python"]'},  # Has ai_tags
                    {"source_path": "/docs/file3.md"},  # No ai_tags
                ],
                documents=["Text 1", "Text 2", "Text 3"],
            ),
        )

        untagged = get_untagged_chunks(provider, batch_size=10)

        # Should only get chunk1 and chunk3
        assert len(untagged) == 2
        assert untagged[0]["id"] == "chunk1"
        assert untagged[1]["id"] == "chunk3"

    def test_get_untagged_chunks_filters_empty_tags(self):
        """Test that chunks with empty ai_tags are considered untagged."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1", "chunk2"],
                metadatas=[
                    {"source_path": "/docs/file1.md", "ai_tags": ""},  # Empty string
                    {"source_path": "/docs/file2.md", "ai_tags": '["python"]'},  # Has tags
                ],
                documents=["Text 1", "Text 2"],
            ),
        )

        untagged = get_untagged_chunks(provider, batch_size=10)

        # chunk1 should be considered untagged
        assert len(untagged) == 1
        assert untagged[0]["id"] == "chunk1"

    def test_get_untagged_chunks_respects_batch_size(self):
        """Test that batch_size limits results."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=[f"chunk{i}" for i in range(100)],
                metadatas=[{"source_path": f"/docs/file{i}.md"} for i in range(100)],
                documents=[f"Text {i}" for i in range(100)],
            ),
        )

        untagged = get_untagged_chunks(provider, batch_size=10)

        # Should stop at batch_size
        assert len(untagged) == 10

    def test_get_untagged_chunks_with_processed_ids(self):
        """Test that processed_ids are skipped."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1", "chunk2", "chunk3"],
                metadatas=[
                    {"source_path": "/docs/file1.md"},
                    {"source_path": "/docs/file2.md"},
                    {"source_path": "/docs/file3.md"},
                ],
                documents=["Text 1", "Text 2", "Text 3"],
            ),
        )

        processed = {"chunk1", "chunk3"}
        untagged = get_untagged_chunks(provider, batch_size=10, processed_ids=processed)

        # Should only get chunk2
        assert len(untagged) == 1
        assert untagged[0]["id"] == "chunk2"

    def test_get_untagged_chunks_fetches_more_than_batch(self):
        """Test that fetch_limit is larger to find enough untagged chunks."""
        provider = _mock_provider()

        untagged = get_untagged_chunks(provider, batch_size=100)

        # Should fetch batch_size * 10 or 1000, whichever is larger
        call_kwargs = provider.get.call_args[1]
        assert call_kwargs["limit"] >= 1000

    def test_get_untagged_chunks_empty(self):
        """Test when no untagged chunks exist."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        untagged = get_untagged_chunks(provider, batch_size=100)

        assert untagged == []

    def test_get_untagged_chunks_error_handling(self):
        """Test graceful error handling."""
        provider = _mock_provider()
        provider.get.side_effect = Exception("Database error")

        untagged = get_untagged_chunks(provider, batch_size=100)

        # Should return empty list on error
        assert untagged == []


class TestTagChunks:
    """Test batch tagging of chunks."""

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_basic(self, mock_get_untagged, mock_generate):
        """Test basic chunk tagging with JSON serialization."""
        # Mock untagged chunks
        mock_get_untagged.side_effect = [
            [
                {
                    "id": "chunk1",
                    "text": "Python tutorial",
                    "metadata": {"source_path": "/docs/file1.md", "chunk_index": 0},
                }
            ],
            [],  # Second call returns empty (done)
        ]

        # Mock tag generation
        mock_generate.return_value = ["python", "tutorial"]

        # Mock provider
        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        # LLM config
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        # Tag chunks
        tagged, failed = tag_chunks(
            provider=provider,
            llm_config=llm_config,
            num_tags=3,
            batch_size=10,
            max_chunks=None,
            dry_run=False,
        )

        assert tagged == 1
        assert failed == 0

        # Verify update_metadatas was called with serialized tags
        provider.update_metadatas.assert_called_once()
        call_kwargs = provider.update_metadatas.call_args[1]
        updated_metadata = call_kwargs["metadatas"][0]

        # Tags should be JSON string
        assert "ai_tags" in updated_metadata
        assert isinstance(updated_metadata["ai_tags"], str)

        # Deserialize and check
        tags = json.loads(updated_metadata["ai_tags"])
        assert "python" in tags
        assert "tutorial" in tags

        # Should have timestamp
        assert "ai_tagged_at" in updated_metadata

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_dry_run(self, mock_get_untagged, mock_generate):
        """Test dry run doesn't update database."""
        mock_get_untagged.side_effect = [
            [
                {
                    "id": "chunk1",
                    "text": "Test",
                    "metadata": {"chunk_index": 0},
                }
            ],
            [],
        ]

        mock_generate.return_value = ["test"]

        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            provider=provider,
            llm_config=llm_config,
            dry_run=True,
        )

        # Should count but not update
        assert tagged == 1
        provider.update_metadatas.assert_not_called()

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_max_limit(self, mock_get_untagged, mock_generate):
        """Test max_chunks limit is respected."""
        # Return more chunks than max
        mock_get_untagged.side_effect = [
            [
                {"id": f"chunk{i}", "text": f"Text {i}", "metadata": {"chunk_index": i}}
                for i in range(5)
            ],
            [],
        ]

        mock_generate.return_value = ["test"]

        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            provider=provider,
            llm_config=llm_config,
            max_chunks=3,  # Limit to 3
            dry_run=False,
        )

        # Should only tag 3
        assert tagged == 3

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_handles_failures(self, mock_get_untagged, mock_generate):
        """Test that failures are counted correctly."""
        mock_get_untagged.side_effect = [
            [
                {"id": "chunk1", "text": "Test 1", "metadata": {"chunk_index": 0}},
                {"id": "chunk2", "text": "Test 2", "metadata": {"chunk_index": 1}},
            ],
            [],
        ]

        # First succeeds, second fails
        mock_generate.side_effect = [
            ["tag1"],
            Exception("Generation failed")
        ]

        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            provider=provider,
            llm_config=llm_config,
            dry_run=False,
        )

        assert tagged == 1
        assert failed == 1

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_uses_existing_tags(self, mock_get_untagged, mock_generate):
        """Test that existing tags are provided as context."""
        mock_get_untagged.side_effect = [
            [
                {"id": "chunk1", "text": "Test", "metadata": {"chunk_index": 0}},
            ],
            [],
        ]

        mock_generate.return_value = ["new-tag"]

        # Provider returns existing tags in the sample query
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["x"],
                metadatas=[{"ai_tags": '["existing-tag", "python"]'}],
            ),
        )

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5-nano",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            provider=provider,
            llm_config=llm_config,
            dry_run=False,
        )

        # Verify generate_tags_for_chunk was called with existing_tags
        call_kwargs = mock_generate.call_args[1]
        assert "existing_tags" in call_kwargs
        assert "existing-tag" in call_kwargs["existing_tags"]
        assert "python" in call_kwargs["existing_tags"]


class TestUserTagManagement:
    """Test user tag add/remove operations with JSON serialization."""

    def test_add_user_tags_new(self):
        """Test adding user tags to chunk without existing tags."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1"],
                metadatas=[{"source_path": "/docs/file.md"}],
            ),
        )

        add_user_tags(provider, "chunk1", ["important", "review"])

        # Verify update_metadatas called with correct metadata
        provider.update_metadatas.assert_called_once()
        call_args = provider.update_metadatas.call_args
        metadata = call_args[1]["metadatas"][0]

        # Tags should be JSON string
        assert "user_tags" in metadata
        assert isinstance(metadata["user_tags"], str)

        # Deserialize and verify
        tags = json.loads(metadata["user_tags"])
        assert "important" in tags
        assert "review" in tags

        assert "user_tagged_at" in metadata

    def test_add_user_tags_existing(self):
        """Test adding tags to chunk with existing user tags."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1"],
                metadatas=[{"user_tags": '["existing"]'}],
            ),
        )

        add_user_tags(provider, "chunk1", ["new"])

        call_args = provider.update_metadatas.call_args
        metadata = call_args[1]["metadatas"][0]

        # Deserialize and check both old and new
        tags = json.loads(metadata["user_tags"])
        assert "existing" in tags
        assert "new" in tags

    def test_add_user_tags_duplicates(self):
        """Test that duplicate tags are avoided."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1"],
                metadatas=[{"user_tags": '["tag1"]'}],
            ),
        )

        add_user_tags(provider, "chunk1", ["tag1", "tag2"])

        call_args = provider.update_metadatas.call_args
        metadata = call_args[1]["metadatas"][0]

        # Deserialize and verify no duplicates
        tags = json.loads(metadata["user_tags"])
        assert tags.count("tag1") == 1
        assert "tag2" in tags

    def test_add_user_tags_normalizes_case(self):
        """Test that tags are normalized to lowercase."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1"],
                metadatas=[{}],
            ),
        )

        add_user_tags(provider, "chunk1", ["Python", "TUTORIAL"])

        call_args = provider.update_metadatas.call_args
        metadata = call_args[1]["metadatas"][0]

        tags = json.loads(metadata["user_tags"])
        assert "python" in tags
        assert "tutorial" in tags

    def test_remove_user_tags(self):
        """Test removing user tags."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["chunk1"],
                metadatas=[{"user_tags": '["tag1", "tag2", "tag3"]'}],
            ),
        )

        remove_user_tags(provider, "chunk1", ["tag2"])

        call_args = provider.update_metadatas.call_args
        metadata = call_args[1]["metadatas"][0]

        tags = json.loads(metadata["user_tags"])
        assert "tag1" in tags
        assert "tag2" not in tags
        assert "tag3" in tags

    def test_add_user_tags_chunk_not_found(self):
        """Test adding tags to non-existent chunk raises error."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        with pytest.raises(ValueError, match="Chunk not found"):
            add_user_tags(provider, "nonexistent", ["tag"])

    def test_remove_user_tags_chunk_not_found(self):
        """Test removing tags from non-existent chunk raises error."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(ids=[], metadatas=[]),
        )

        with pytest.raises(ValueError, match="Chunk not found"):
            remove_user_tags(provider, "nonexistent", ["tag"])


class TestGetAllTags:
    """Test retrieving all tags from collection."""

    def test_get_all_tags_both_types(self):
        """Test getting both AI and user tags with JSON deserialization."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1", "2"],
                metadatas=[
                    {
                        "ai_tags": '["python", "tutorial"]',
                        "user_tags": '["important"]',
                    },
                    {
                        "ai_tags": '["python", "advanced"]',
                        "user_tags": '["review"]',
                    },
                ],
            ),
        )

        all_tags = get_all_tags(provider)

        assert "ai_tags" in all_tags
        assert "user_tags" in all_tags

        # Check for unique tags
        assert "python" in all_tags["ai_tags"]
        assert "tutorial" in all_tags["ai_tags"]
        assert "advanced" in all_tags["ai_tags"]

        assert "important" in all_tags["user_tags"]
        assert "review" in all_tags["user_tags"]

    def test_get_all_tags_empty(self):
        """Test when no tags exist."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1"],
                metadatas=[{"source_path": "/docs/file.md"}],
            ),
        )

        all_tags = get_all_tags(provider)

        assert all_tags["ai_tags"] == []
        assert all_tags["user_tags"] == []

    def test_get_all_tags_deduplication(self):
        """Test that duplicate tags are removed."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1", "2", "3"],
                metadatas=[
                    {"ai_tags": '["python", "tutorial"]'},
                    {"ai_tags": '["python", "advanced"]'},
                    {"ai_tags": '["python"]'},
                ],
            ),
        )

        all_tags = get_all_tags(provider)

        # python should only appear once
        assert all_tags["ai_tags"].count("python") == 1

    def test_get_all_tags_sorted(self):
        """Test that returned tags are sorted."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1"],
                metadatas=[{"ai_tags": '["zebra", "apple", "monkey"]'}],
            ),
        )

        all_tags = get_all_tags(provider)

        # Should be alphabetically sorted
        assert all_tags["ai_tags"] == ["apple", "monkey", "zebra"]

    def test_get_all_tags_handles_invalid_json(self):
        """Test that invalid JSON is handled gracefully."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1", "2"],
                metadatas=[
                    {"ai_tags": "invalid json"},
                    {"ai_tags": '["valid"]'},
                ],
            ),
        )

        all_tags = get_all_tags(provider)

        # Should only get valid tags
        assert "valid" in all_tags["ai_tags"]
        assert len(all_tags["ai_tags"]) == 1

    def test_get_all_tags_ignores_non_json_values(self):
        """Non-JSON tag metadata should be ignored."""
        provider = _mock_provider(
            get_result=VectorDBGetResult(
                ids=["1", "2"],
                metadatas=[
                    {"ai_tags": ["legacy", "list"]},
                    {"ai_tags": '["new", "json"]'},
                ],
            ),
        )

        all_tags = get_all_tags(provider)

        assert "legacy" not in all_tags["ai_tags"]
        assert "list" not in all_tags["ai_tags"]
        assert "new" in all_tags["ai_tags"]
        assert "json" in all_tags["ai_tags"]
