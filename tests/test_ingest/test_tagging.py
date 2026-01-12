"""
Tests for AI tagging module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from lsm.ingest.tagging import (
    generate_tags_for_chunk,
    get_untagged_chunks,
    tag_chunks,
    add_user_tags,
    remove_user_tags,
    get_all_tags,
)
from lsm.config.models import LLMConfig


class TestTagGeneration:
    """Test tag generation from LLM."""

    @patch("lsm.ingest.tagging.OpenAI")
    def test_generate_tags_basic(self, mock_openai_class):
        """Test basic tag generation."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["python", "programming", "tutorial"]'
        mock_client.chat.completions.create.return_value = mock_response

        # Create LLM config
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
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

    @patch("lsm.ingest.tagging.OpenAI")
    def test_generate_tags_with_existing_context(self, mock_openai_class):
        """Test tag generation with existing tags context."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["machine-learning", "python"]'
        mock_client.chat.completions.create.return_value = mock_response

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
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
        assert all(isinstance(t, str) for t in tags)

    @patch("lsm.ingest.tagging.OpenAI")
    def test_generate_tags_non_json_response(self, mock_openai_class):
        """Test handling of non-JSON LLM response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Comma-separated response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "python, programming, tutorial"
        mock_client.chat.completions.create.return_value = mock_response

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
        )

        tags = generate_tags_for_chunk(
            text="Python tutorial",
            llm_config=llm_config,
            num_tags=3,
        )

        assert len(tags) == 3
        assert "python" in tags
        assert "programming" in tags


class TestUntaggedChunks:
    """Test finding untagged chunks."""

    def test_get_untagged_chunks_basic(self):
        """Test getting untagged chunks."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "metadatas": [
                {"source_path": "/docs/file1.md"},
                {"source_path": "/docs/file2.md"},
            ],
            "documents": ["Text 1", "Text 2"],
        }

        untagged = get_untagged_chunks(mock_collection, batch_size=10)

        assert len(untagged) == 2
        assert untagged[0]["id"] == "chunk1"
        assert untagged[0]["text"] == "Text 1"
        assert untagged[1]["id"] == "chunk2"

    def test_get_untagged_chunks_with_where_clause(self):
        """Test that where clause is used correctly."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk1"],
            "metadatas": [{"source_path": "/docs/file1.md"}],
            "documents": ["Text 1"],
        }

        untagged = get_untagged_chunks(mock_collection, batch_size=100)

        # Verify where clause was passed
        mock_collection.get.assert_called_once()
        call_args = mock_collection.get.call_args
        assert "where" in call_args[1]

    def test_get_untagged_chunks_empty(self):
        """Test when no untagged chunks exist."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
            "documents": [],
        }

        untagged = get_untagged_chunks(mock_collection, batch_size=100)

        assert untagged == []


class TestTagChunks:
    """Test batch tagging of chunks."""

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_basic(self, mock_get_untagged, mock_generate):
        """Test basic chunk tagging."""
        # Mock untagged chunks
        mock_get_untagged.side_effect = [
            [
                {
                    "id": "chunk1",
                    "text": "Python tutorial",
                    "metadata": {"source_path": "/docs/file1.md"},
                }
            ],
            [],  # Second call returns empty (done)
        ]

        # Mock tag generation
        mock_generate.return_value = ["python", "tutorial"]

        # Mock collection
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": []
        }

        # LLM config
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
        )

        # Tag chunks
        tagged, failed = tag_chunks(
            collection=mock_collection,
            llm_config=llm_config,
            num_tags=3,
            batch_size=10,
            max_chunks=None,
            dry_run=False,
        )

        assert tagged == 1
        assert failed == 0

        # Verify update was called
        mock_collection.update.assert_called_once()

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_dry_run(self, mock_get_untagged, mock_generate):
        """Test dry run doesn't update database."""
        mock_get_untagged.side_effect = [
            [
                {
                    "id": "chunk1",
                    "text": "Test",
                    "metadata": {},
                }
            ],
            [],
        ]

        mock_generate.return_value = ["test"]

        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            collection=mock_collection,
            llm_config=llm_config,
            dry_run=True,
        )

        # Should count but not update
        assert tagged == 1
        mock_collection.update.assert_not_called()

    @patch("lsm.ingest.tagging.generate_tags_for_chunk")
    @patch("lsm.ingest.tagging.get_untagged_chunks")
    def test_tag_chunks_max_limit(self, mock_get_untagged, mock_generate):
        """Test max_chunks limit is respected."""
        # Return more chunks than max
        mock_get_untagged.side_effect = [
            [
                {"id": f"chunk{i}", "text": f"Text {i}", "metadata": {}}
                for i in range(5)
            ],
            [],
        ]

        mock_generate.return_value = ["test"]

        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        llm_config = LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
        )

        tagged, failed = tag_chunks(
            collection=mock_collection,
            llm_config=llm_config,
            max_chunks=3,  # Limit to 3
            dry_run=False,
        )

        # Should only tag 3
        assert tagged == 3


class TestUserTagManagement:
    """Test user tag add/remove operations."""

    def test_add_user_tags_new(self):
        """Test adding user tags to chunk without existing tags."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [{"source_path": "/docs/file.md"}]
        }

        add_user_tags(mock_collection, "chunk1", ["important", "review"])

        # Verify update called with correct metadata
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]

        assert "user_tags" in metadata
        assert "important" in metadata["user_tags"]
        assert "review" in metadata["user_tags"]
        assert "user_tagged_at" in metadata

    def test_add_user_tags_existing(self):
        """Test adding tags to chunk with existing user tags."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [{"user_tags": ["existing"]}]
        }

        add_user_tags(mock_collection, "chunk1", ["new"])

        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]

        # Should have both old and new
        assert "existing" in metadata["user_tags"]
        assert "new" in metadata["user_tags"]

    def test_add_user_tags_duplicates(self):
        """Test that duplicate tags are avoided."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [{"user_tags": ["tag1"]}]
        }

        add_user_tags(mock_collection, "chunk1", ["tag1", "tag2"])

        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]

        # Should only have one instance of tag1
        assert metadata["user_tags"].count("tag1") == 1
        assert "tag2" in metadata["user_tags"]

    def test_remove_user_tags(self):
        """Test removing user tags."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [{"user_tags": ["tag1", "tag2", "tag3"]}]
        }

        remove_user_tags(mock_collection, "chunk1", ["tag2"])

        call_args = mock_collection.update.call_args
        metadata = call_args[1]["metadatas"][0]

        assert "tag1" in metadata["user_tags"]
        assert "tag2" not in metadata["user_tags"]
        assert "tag3" in metadata["user_tags"]

    def test_add_user_tags_chunk_not_found(self):
        """Test adding tags to non-existent chunk raises error."""
        mock_collection = Mock()
        mock_collection.get.return_value = {"metadatas": []}

        with pytest.raises(ValueError, match="Chunk not found"):
            add_user_tags(mock_collection, "nonexistent", ["tag"])


class TestGetAllTags:
    """Test retrieving all tags from collection."""

    def test_get_all_tags_both_types(self):
        """Test getting both AI and user tags."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "ai_tags": ["python", "tutorial"],
                    "user_tags": ["important"],
                },
                {
                    "ai_tags": ["python", "advanced"],
                    "user_tags": ["review"],
                },
            ]
        }

        all_tags = get_all_tags(mock_collection)

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
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_path": "/docs/file.md"},
            ]
        }

        all_tags = get_all_tags(mock_collection)

        assert all_tags["ai_tags"] == []
        assert all_tags["user_tags"] == []

    def test_get_all_tags_deduplication(self):
        """Test that duplicate tags are removed."""
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"ai_tags": ["python", "tutorial"]},
                {"ai_tags": ["python", "advanced"]},
                {"ai_tags": ["python"]},
            ]
        }

        all_tags = get_all_tags(mock_collection)

        # python should only appear once
        assert all_tags["ai_tags"].count("python") == 1
