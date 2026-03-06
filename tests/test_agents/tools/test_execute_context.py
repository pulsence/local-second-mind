"""
Tests for ExecuteContextTool.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lsm.agents.tools.execute_context import ExecuteContextTool
from lsm.query.pipeline_types import (
    ContextPackage,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
)


# ---------------------------------------------------------------------------
# Fake pipeline
# ---------------------------------------------------------------------------


class _FakeConfig:
    def get_mode_config(self, name):
        from lsm.config.models.modes import GROUNDED_MODE
        return GROUNDED_MODE


class FakePipeline:
    """Minimal stand-in for RetrievalPipeline stages 2+3."""

    def __init__(self, answer="The answer is 42.", response_id=None):
        self._answer = answer
        self._response_id = response_id
        self.last_package = None
        self.config = _FakeConfig()

    def synthesize_context(self, package):
        # Just populate context_block for the execute stage
        return replace(
            package,
            context_block=package.context_block or "context",
            source_labels={"S1": {"source_path": "/doc.md"}},
        )

    def execute(self, package):
        self.last_package = package
        return QueryResponse(
            answer=self._answer,
            package=package,
            conversation_id=package.request.conversation_id,
            response_id=self._response_id,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_serialized_query_response():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({
        "question": "What is Python?",
        "context_block": "[S1] Python is a programming language.",
    }))

    assert output["answer"] == "The answer is 42."


def test_question_required():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    with pytest.raises(ValueError, match="question is required"):
        tool.execute({"question": "", "context_block": "text"})


def test_context_block_required():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    with pytest.raises(ValueError, match="context_block is required"):
        tool.execute({"question": "test", "context_block": ""})


def test_starting_prompt_forwarded():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    tool.execute({
        "question": "test",
        "context_block": "context text",
        "starting_prompt": "Custom instructions",
    })

    assert pipeline.last_package.starting_prompt is not None


def test_response_id_in_output():
    pipeline = FakePipeline(response_id="resp-xyz")
    tool = ExecuteContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({
        "question": "test",
        "context_block": "context text",
    }))

    assert output["response_id"] == "resp-xyz"


def test_conversation_id_forwarded():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    output = json.loads(tool.execute({
        "question": "test",
        "context_block": "context text",
        "conversation_id": "conv-123",
    }))

    assert output.get("conversation_id") == "conv-123"


def test_mode_forwarded():
    pipeline = FakePipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    tool.execute({
        "question": "test",
        "context_block": "context text",
        "mode": "insight",
    })

    assert pipeline.last_package.request.mode == "insight"


def test_explicit_context_block_is_preserved_without_candidates():
    class OverwritingPipeline(FakePipeline):
        def synthesize_context(self, package):
            return replace(package, context_block="")

    pipeline = OverwritingPipeline()
    tool = ExecuteContextTool(pipeline=pipeline)

    tool.execute(
        {
            "question": "test",
            "context_block": "[S1] Preserved context.",
        }
    )

    assert pipeline.last_package.context_block == "[S1] Preserved context."
