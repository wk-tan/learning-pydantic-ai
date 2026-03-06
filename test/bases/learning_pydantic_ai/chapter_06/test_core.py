"""Tests for Chapter 6: Streaming.

Demonstrates:
- run_stream() returns a StreamedRunResult that yields text chunks
- stream_text(delta=True) gives incremental chunks
- stream_text(delta=False) gives cumulative text
- get_output() returns the validated structured output after streaming
- Streaming with tools: tools resolve first, then final answer streams
- all_messages() works identically on streamed results
- FunctionModel needs a stream_function (AsyncIterator) for run_stream()
- TestModel supports streaming natively (uses same _request internally)
"""

from collections.abc import AsyncIterator

import pytest
from learning_pydantic_ai.chapter_06.core import (
    DataSummary,
    structured_agent,
    text_agent,
    tool_agent,
)
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Helpers ---


def _stream_text_model(text: str = "Streamed answer."):
    """FunctionModel with a stream_function that yields text word-by-word."""

    async def stream_handler(messages: list, info: AgentInfo) -> AsyncIterator[str]:
        for word in text.split(" "):
            yield word + " "

    return FunctionModel(stream_function=stream_handler)


# === Test: Basic text streaming ===


@pytest.mark.anyio
async def test_stream_text_delta_true():
    """stream_text(delta=True) yields incremental chunks."""
    with text_agent.override(model=_stream_text_model("Hello world.")):
        async with text_agent.run_stream("Say hello") as result:
            chunks = [chunk async for chunk in result.stream_text(delta=True)]

    combined = "".join(chunks)
    assert "Hello" in combined
    assert "world." in combined


@pytest.mark.anyio
async def test_stream_text_delta_false():
    """stream_text(delta=False) yields cumulative text."""
    with text_agent.override(model=_stream_text_model("Cumulative result here.")):
        async with text_agent.run_stream("Test") as result:
            chunks = [chunk async for chunk in result.stream_text(delta=False)]

    # Each successive chunk should be at least as long as the previous
    for i in range(1, len(chunks)):
        assert len(chunks[i]) >= len(chunks[i - 1])
    # Last chunk contains the full text
    assert "Cumulative" in chunks[-1]
    assert "here." in chunks[-1]


# === Test: Structured output streaming ===


@pytest.mark.anyio
async def test_stream_structured_output():
    """Streaming a structured agent yields valid output via get_output()."""
    with structured_agent.override(
        model=TestModel(
            custom_output_args={
                "dataset_name": "sales",
                "row_count": 4,
                "key_insight": "April had the highest revenue.",
            }
        )
    ):
        async with structured_agent.run_stream("Summarize sales") as result:
            output = await result.get_output()

    assert isinstance(output, DataSummary)
    assert output.dataset_name == "sales"
    assert output.row_count == 4
    assert "April" in output.key_insight


# === Test: Streaming with tools ===


@pytest.mark.anyio
async def test_stream_with_tools():
    """TestModel calls tools then streams the final text answer.

    TestModel with call_tools='all' calls every registered tool on the first
    request, then returns a text answer on the second. Streaming uses the same
    internal _request logic, so tools resolve before the final answer streams.
    """
    with tool_agent.override(model=TestModel()):
        async with tool_agent.run_stream("Tell me about the data") as result:
            chunks = [chunk async for chunk in result.stream_text(delta=True)]

    combined = "".join(chunks)
    assert len(combined) > 0


@pytest.mark.anyio
async def test_stream_with_tools_messages_structure():
    """all_messages() on a streamed result includes tool call/return pairs."""
    with tool_agent.override(model=TestModel()):
        async with tool_agent.run_stream("Tell me about the data") as result:
            async for _ in result.stream_text(delta=True):
                pass

    msgs = result.all_messages()

    # TestModel calls all tools, so we expect:
    # Request(user), Response(tool_calls), Request(tool_returns), Response(text)
    assert len(msgs) >= 4

    tool_calls = [p for msg in msgs for p in msg.parts if isinstance(p, ToolCallPart)]
    tool_returns = [
        p for msg in msgs for p in msg.parts if isinstance(p, ToolReturnPart)
    ]
    # tool_agent has 2 tools: list_datasets, query_dataset
    assert len(tool_calls) >= 1
    assert len(tool_returns) >= 1


# === Test: all_messages() on streamed results ===


@pytest.mark.anyio
async def test_stream_all_messages_has_user_prompt():
    """Streamed result all_messages() contains the original user prompt."""
    with text_agent.override(model=_stream_text_model()):
        async with text_agent.run_stream("What is streaming?") as result:
            async for _ in result.stream_text(delta=True):
                pass

    msgs = result.all_messages()
    assert isinstance(msgs[0], ModelRequest)
    user_parts = [p for p in msgs[0].parts if isinstance(p, UserPromptPart)]
    assert any("What is streaming?" in p.content for p in user_parts)


@pytest.mark.anyio
async def test_stream_all_messages_ends_with_response():
    """Streamed result all_messages() ends with a ModelResponse."""
    with text_agent.override(model=_stream_text_model("Final.")):
        async with text_agent.run_stream("Hi") as result:
            async for _ in result.stream_text(delta=True):
                pass

    msgs = result.all_messages()
    assert isinstance(msgs[-1], ModelResponse)
