"""
Tests for OpenAI provider implementation.
"""

import json
import pytest
from unittest.mock import Mock, patch

from lsm.config.models import LLMConfig
from lsm.providers.openai import OpenAIProvider


class TestOpenAIProvider:
    """Tests for OpenAIProvider implementation."""

    @pytest.fixture
    def llm_config(self):
        """Create test LLM configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=2000,
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = Mock()
        mock_client.responses = Mock()
        mock_client.models = Mock()
        return mock_client

    def test_provider_initialization(self, llm_config):
        """Test OpenAI provider initialization."""
        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(llm_config)

            assert provider.name == "openai"
            assert provider.model == "gpt-5.2"
            mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_provider_initialization_without_api_key(self):
        """Test OpenAI provider uses env var when no API key in config."""
        config = LLMConfig(provider="openai", model="gpt-5.2")

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            provider = OpenAIProvider(config)

            assert provider.name == "openai"
            mock_openai.assert_called_once_with()  # No api_key parameter

    def test_is_available_with_api_key(self, llm_config):
        """Test is_available returns True when API key is set."""
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test is_available returns False when no API key."""
        config = LLMConfig(provider="openai", model="gpt-5.2", api_key=None)

        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            assert provider.is_available() is False

    def test_rerank_with_valid_candidates(self, llm_config):
        """Test rerank with valid candidates."""
        candidates = [
            {
                "text": "Python is a programming language",
                "metadata": {"source_path": "/docs/python.md", "chunk_index": 0},
                "distance": 0.2,
            },
            {
                "text": "Java is also a programming language",
                "metadata": {"source_path": "/docs/java.md", "chunk_index": 0},
                "distance": 0.3,
            },
            {
                "text": "Python has great libraries",
                "metadata": {"source_path": "/docs/python.md", "chunk_index": 1},
                "distance": 0.25,
            },
        ]

        mock_response = Mock()
        mock_response.output_text = json.dumps({
            "ranking": [
                {"index": 0, "reason": "Most relevant"},
                {"index": 2, "reason": "Also relevant"},
            ]
        })

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("What is Python?", candidates, k=2)

            assert len(result) == 2
            assert result[0]["text"] == "Python is a programming language"
            assert result[1]["text"] == "Python has great libraries"

    def test_rerank_with_empty_candidates(self, llm_config):
        """Test rerank with empty candidate list."""
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            result = provider.rerank("What is Python?", [], k=5)

            assert result == []

    def test_rerank_fallback_on_api_error(self, llm_config):
        """Test rerank falls back to original order on API error."""
        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
            {"text": "Third", "metadata": {}, "distance": 0.3},
        ]

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("Question?", candidates, k=2)

            # Should return first k candidates
            assert len(result) == 2
            assert result[0]["text"] == "First"
            assert result[1]["text"] == "Second"

    def test_rerank_fallback_on_invalid_json(self, llm_config):
        """Test rerank falls back on invalid JSON response."""
        candidates = [
            {"text": "First", "metadata": {}, "distance": 0.1},
            {"text": "Second", "metadata": {}, "distance": 0.2},
        ]

        mock_response = Mock()
        mock_response.output_text = "Not valid JSON"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            result = provider.rerank("Question?", candidates, k=2)

            assert len(result) == 2
            assert result[0]["text"] == "First"

    def test_synthesize_grounded_mode(self, llm_config):
        """Test synthesize in grounded mode."""
        mock_response = Mock()
        mock_response.output_text = "Python is a programming language [S1]."

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "What is Python?",
                "[S1] Python is a high-level language",
                mode="grounded",
            )

            assert "[S1]" in answer
            mock_client.responses.create.assert_called_once()

    def test_synthesize_insight_mode(self, llm_config):
        """Test synthesize in insight mode."""
        mock_response = Mock()
        mock_response.output_text = "Analysis of Python ecosystem [S1] [S2]."

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "Analyze Python",
                "[S1] Python docs\n[S2] Python guide",
                mode="insight",
            )

            assert "Analysis" in answer
            mock_client.responses.create.assert_called_once()

    def test_synthesize_fallback_on_error(self, llm_config):
        """Test synthesize returns fallback on API error."""
        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "What is Python?",
                "[S1] Context here",
                mode="grounded",
            )

            assert "Offline mode" in answer
            assert "What is Python?" in answer

    def test_synthesize_includes_server_cache_args_and_tracks_response_id(self, llm_config):
        """Test OpenAI server-side cache args are forwarded and response id captured."""
        mock_response = Mock()
        mock_response.output_text = "Cached answer [S1]."
        mock_response.id = "resp_123"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            answer = provider.synthesize(
                "Follow-up?",
                "[S1] Context",
                mode="grounded",
                enable_server_cache=True,
                previous_response_id="resp_prev",
                prompt_cache_key="openai:gpt-5.2:grounded",
            )

            assert "Cached answer" in answer
            assert provider.last_response_id == "resp_123"
            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["previous_response_id"] == "resp_prev"
            assert kwargs["prompt_cache_key"] == "openai:gpt-5.2:grounded"

    def test_generate_tags_basic(self, llm_config):
        """Test basic tag generation."""
        mock_response = Mock()
        mock_response.output_text = '{"tags": ["python", "programming", "tutorial"]}'

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            tags = provider.generate_tags("This is a Python programming tutorial", num_tags=3)

            assert len(tags) == 3
            assert "python" in tags
            assert "programming" in tags
            assert "tutorial" in tags

    def test_generate_tags_with_existing_context(self, llm_config):
        """Test tag generation with existing tags context."""
        mock_response = Mock()
        mock_response.output_text = '{"tags": ["machine-learning", "python"]}'

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            existing_tags = ["python", "data-science", "tutorial"]

            tags = provider.generate_tags(
                "Machine learning with Python",
                num_tags=2,
                existing_tags=existing_tags
            )

            assert len(tags) <= 2
            assert all(isinstance(t, str) for t in tags)

    def test_generate_tags_empty_response(self, llm_config):
        """Test handling of empty LLM response."""
        mock_response = Mock()
        mock_response.output_text = ""

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            tags = provider.generate_tags("Test text")

            assert tags == []

    def test_generate_tags_json_extraction(self, llm_config):
        """Test JSON extraction from markdown-wrapped response."""
        mock_response = Mock()
        # Response with markdown code block
        mock_response.output_text = '```json\n{"tags": ["python", "code"]}\n```'

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            tags = provider.generate_tags("Python code")

            assert len(tags) == 2
            assert "python" in tags
            assert "code" in tags

    def test_generate_tags_comma_separated_fallback(self, llm_config):
        """Test fallback to comma-separated parsing."""
        mock_response = Mock()
        # Response without JSON structure
        mock_response.output_text = "python, programming, tutorial"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            tags = provider.generate_tags("Python tutorial", num_tags=3)

            assert len(tags) == 3
            assert "python" in tags
            assert "programming" in tags
            assert "tutorial" in tags

    def test_estimate_cost_gpt5(self, llm_config):
        """Test cost estimation for GPT-5 model."""
        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(llm_config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

            assert cost is not None
            assert cost > 0
            assert isinstance(cost, float)

    def test_estimate_cost_gpt4(self):
        """Test cost estimation for GPT-4 model."""
        config = LLMConfig(provider="openai", model="gpt-4", api_key="test")

        with patch("lsm.providers.openai.OpenAI"):
            provider = OpenAIProvider(config)
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)

            assert cost is not None
            assert cost > 0

    def test_send_message_uses_plain_string_input_and_optional_instruction(self, llm_config):
        """OpenAI payload should use string input and omit instructions when None."""
        mock_response = Mock()
        mock_response.output_text = "ok"
        mock_response.id = "resp_plain"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            provider.send_message(input="hello", instruction=None, prompt="preface", max_tokens=20)

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["input"] == "preface\n\nhello"
            assert "instructions" not in kwargs

    def test_send_message_forwards_cache_params_without_enable_gate(self, llm_config):
        """Cache params should be forwarded even without enable_server_cache."""
        mock_response = Mock()
        mock_response.output_text = "cached"
        mock_response.id = "resp_new"

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = mock_response
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            provider.send_message(
                input="follow-up",
                instruction="system",
                previous_response_id="resp_prev",
                prompt_cache_key="key-1",
                prompt_cache_retention=7200,
                max_tokens=40,
            )

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["previous_response_id"] == "resp_prev"
            assert kwargs["prompt_cache_key"] == "key-1"
            assert kwargs["prompt_cache_retention"] == 7200
            assert provider.last_response_id == "resp_new"

    def test_streaming_supports_cache_prompt_tools_and_json_schema(self, llm_config):
        """Streaming call should include parity params and payload features."""
        stream_events = [
            {"type": "response.output_text.delta", "delta": "hello"},
            {"type": "response.completed", "response": {"id": "resp_stream"}},
        ]

        with patch("lsm.providers.openai.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_client.responses.create.return_value = iter(stream_events)
            mock_openai.return_value = mock_client

            provider = OpenAIProvider(llm_config)
            chunks = list(
                provider.send_streaming_message(
                    input="payload",
                    instruction="rules",
                    prompt="prefix",
                    previous_response_id="resp_prev",
                    prompt_cache_key="cache-key",
                    prompt_cache_retention=600,
                    json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    tools=[
                        {
                            "name": "ping",
                            "description": "Ping tool",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    tool_choice="auto",
                    max_tokens=50,
                )
            )

            assert "".join(chunks) == "hello"
            assert provider.last_response_id == "resp_stream"

            kwargs = mock_client.responses.create.call_args.kwargs
            assert kwargs["input"] == "prefix\n\npayload"
            assert kwargs["previous_response_id"] == "resp_prev"
            assert kwargs["prompt_cache_key"] == "cache-key"
            assert kwargs["prompt_cache_retention"] == 600
            assert kwargs["tools"][0]["function"]["name"] == "ping"
            assert kwargs["tool_choice"] == "auto"
            assert kwargs["text"]["format"]["type"] == "json_schema"
            assert kwargs["stream"] is True
