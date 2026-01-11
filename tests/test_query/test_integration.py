"""
Integration tests for query module.

Tests the full query flow from embedding to synthesis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from lsm.config.models import LSMConfig, LLMConfig, QueryConfig
from lsm.query.session import Candidate


class TestQueryIntegration:
    """Integration tests for complete query flow."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return LSMConfig(
            persist_dir=Path("/test/.chroma"),
            collection="test_kb",
            embed_model="test-model",
            device="cpu",
            batch_size=32,
            llm=LLMConfig(
                provider="openai",
                model="gpt-5.2",
                api_key="test-key",
            ),
            query=QueryConfig(
                k=12,
                k_rerank=6,
                no_rerank=False,
                mode="grounded",
            ),
        )

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        return embedder

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        collection = Mock()
        collection.count.return_value = 100
        collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [[
                "Python is a programming language",
                "Python has great libraries",
                "Python supports OOP",
            ]],
            "metadatas": [[
                {"source_path": "/docs/python.md", "chunk_index": 0, "ext": ".md"},
                {"source_path": "/docs/python.md", "chunk_index": 1, "ext": ".md"},
                {"source_path": "/docs/python.md", "chunk_index": 2, "ext": ".md"},
            ]],
            "distances": [[0.1, 0.15, 0.2]],
        }
        return collection

    @patch("lsm.query.retrieval.init_collection")
    @patch("lsm.query.retrieval.init_embedder")
    @patch("lsm.query.providers.openai.OpenAI")
    def test_full_query_flow_no_rerank(
        self,
        mock_openai_class,
        mock_init_embedder,
        mock_init_collection,
        mock_config,
        mock_embedder,
        mock_collection,
    ):
        """Test complete query flow without LLM reranking."""
        from lsm.query.cli import run_query_cli

        # Setup mocks
        mock_init_embedder.return_value = mock_embedder
        mock_init_collection.return_value = mock_collection

        mock_openai_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Python is a programming language [S1]."
        mock_openai_client.responses.create.return_value = mock_response
        mock_openai_class.return_value = mock_openai_client

        # Disable reranking
        mock_config.query.no_rerank = True

        # Mock input to exit immediately
        with patch("builtins.input", side_effect=["What is Python?", "/exit"]):
            with patch("builtins.print"):  # Suppress output
                result = run_query_cli(mock_config)

        # Should complete successfully
        assert result == 0

    @patch("lsm.query.providers.openai.OpenAI")
    def test_provider_rerank_integration(self, mock_openai_class):
        """Test provider rerank integration."""
        from lsm.query.providers import create_provider

        config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test")

        # Mock rerank response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = '{"ranking": [{"index": 2, "reason": "Best"}, {"index": 0, "reason": "Good"}]}'
        mock_client.responses.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = create_provider(config)

        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
            {"text": "Third", "metadata": {}, "distance": 0.3},
        ]

        result = provider.rerank("Question?", candidates, k=2)

        assert len(result) == 2
        assert result[0]["text"] == "Third"  # Reranked to first
        assert result[1]["text"] == "First"  # Reranked to second

    @patch("lsm.query.providers.openai.OpenAI")
    def test_provider_synthesize_integration(self, mock_openai_class):
        """Test provider synthesize integration."""
        from lsm.query.providers import create_provider

        config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test")

        # Mock synthesize response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.output_text = "Python is a programming language [S1]."
        mock_client.responses.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = create_provider(config)

        context = "[S1] Python docs"
        answer = provider.synthesize("What is Python?", context, mode="grounded")

        assert "[S1]" in answer
        assert "programming language" in answer

    def test_local_reranking_pipeline_integration(self):
        """Test local reranking pipeline integration."""
        from lsm.query.rerank import apply_local_reranking

        candidates = [
            Candidate(
                cid="1",
                text="Python programming language",
                meta={"source_path": "/docs/python.md"},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Python programming language",  # Duplicate
                meta={"source_path": "/docs/python_copy.md"},
                distance=0.15,
            ),
            Candidate(
                cid="3",
                text="Python tutorials",
                meta={"source_path": "/docs/python.md"},
                distance=0.2,
            ),
            Candidate(
                cid="4",
                text="Python guides",
                meta={"source_path": "/docs/python.md"},
                distance=0.25,
            ),
        ]

        result = apply_local_reranking(
            "Python programming",
            candidates,
            max_per_file=2,
            local_pool=10,
        )

        # Should deduplicate
        texts = [c.text for c in result]
        assert len(texts) == len(set(texts))

        # Should enforce diversity (max 2 per file)
        from_python_md = [c for c in result if c.source_path == "/docs/python.md"]
        assert len(from_python_md) <= 2

    def test_retrieval_and_filtering_integration(self):
        """Test retrieval and filtering integration."""
        from lsm.query.retrieval import filter_candidates
        from lsm.query.session import Candidate

        candidates = [
            Candidate(
                cid="1",
                text="Python text",
                meta={"source_path": "/docs/python.md", "ext": ".md"},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Java text",
                meta={"source_path": "/docs/java.pdf", "ext": ".pdf"},
                distance=0.15,
            ),
            Candidate(
                cid="3",
                text="Python guide",
                meta={"source_path": "/guides/python.txt", "ext": ".txt"},
                distance=0.2,
            ),
        ]

        # Filter for Python files with .md or .txt
        result = filter_candidates(
            candidates,
            path_contains="python",
            ext_allow=[".md", ".txt"],
        )

        assert len(result) == 2
        assert all("python" in c.source_path.lower() for c in result)
        assert all(c.ext in [".md", ".txt"] for c in result)

    def test_synthesis_and_formatting_integration(self):
        """Test synthesis and formatting integration."""
        from lsm.query.synthesis import build_context_block, format_source_list

        candidates = [
            Candidate(
                cid="1",
                text="Python is a language",
                meta={"source_path": "/docs/python.md", "source_name": "python.md", "chunk_index": 0},
                distance=0.1,
            ),
            Candidate(
                cid="2",
                text="Python has libraries",
                meta={"source_path": "/docs/python.md", "source_name": "python.md", "chunk_index": 1},
                distance=0.15,
            ),
        ]

        # Build context
        context, sources = build_context_block(candidates)

        assert "[S1]" in context
        assert "[S2]" in context
        assert "Python is a language" in context

        # Format sources
        formatted = format_source_list(sources)

        assert "Sources:" in formatted
        assert "[S1] [S2]" in formatted or ("[S1]" in formatted and "[S2]" in formatted)
        assert "python.md" in formatted


class TestErrorHandling:
    """Integration tests for error handling."""

    @patch("lsm.query.retrieval.init_collection")
    @patch("lsm.query.retrieval.init_embedder")
    def test_query_with_empty_collection(
        self,
        mock_init_embedder,
        mock_init_collection,
    ):
        """Test query flow with empty collection."""
        from lsm.query.cli import run_query_cli
        from lsm.config.models import LSMConfig, LLMConfig, QueryConfig

        config = LSMConfig(
            persist_dir=Path("/test/.chroma"),
            collection="empty_kb",
            llm=LLMConfig(provider="openai", model="gpt-5.2", api_key="test"),
            query=QueryConfig(),
        )

        # Mock empty collection
        mock_embedder = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 0

        mock_init_embedder.return_value = mock_embedder
        mock_init_collection.return_value = mock_collection

        with patch("lsm.query.providers.openai.OpenAI"):
            result = run_query_cli(config)

        # Should exit with error code
        assert result == 1

    @patch("lsm.query.providers.openai.OpenAI")
    def test_provider_graceful_degradation(self, mock_openai_class):
        """Test provider gracefully degrades on API errors."""
        from lsm.query.providers import create_provider

        config = LLMConfig(provider="openai", model="gpt-5.2", api_key="test")

        # Mock API error
        mock_client = Mock()
        mock_client.responses.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        provider = create_provider(config)

        # Rerank should fall back to original order
        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
        ]

        result = provider.rerank("Question?", candidates, k=2)
        assert len(result) == 2
        assert result[0]["text"] == "First"

        # Synthesize should return fallback answer
        answer = provider.synthesize("Question?", "Context", mode="grounded")
        assert "Offline mode" in answer
