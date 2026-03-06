"""Tests for Chapter 5: Message History & Conversational State.

Demonstrates:
- all_messages() captures the full conversation including tool calls
- message_history= replays prior context into a new run
- capture_run_messages() captures only the current run's messages
- Message part types: UserPromptPart, TextPart, ToolCallPart, ToolReturnPart
- usage() returns token consumption per run
- trim_history preserves tool call pairs
"""

from learning_pydantic_ai.chapter_05.core import (
    explorer_agent,
    trim_history,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Helpers ---


def _model_list_then_answer():
    """FunctionModel that calls list_datasets, then gives a text answer."""
    call_count = 0

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="list_datasets", args={})]
            )
        return ModelResponse(
            parts=[TextPart(content="Available datasets: sales, products.")]
        )

    return FunctionModel(handler)


def _model_query_sales():
    """FunctionModel that calls query_dataset('sales'), then answers."""
    call_count = 0

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_dataset",
                        args={"dataset_name": "sales"},
                    )
                ]
            )
        return ModelResponse(
            parts=[TextPart(content="The sales data shows 4 months of records.")]
        )

    return handler


def _model_text_only(text: str = "Here is my answer."):
    """FunctionModel that returns a plain text answer (no tool calls)."""

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(handler)


# --- Test: all_messages() structure ---


def test_all_messages_contains_request_and_response():
    """all_messages() returns alternating ModelRequest/ModelResponse."""
    with explorer_agent.override(model=_model_text_only()):
        result = explorer_agent.run_sync("Hello")

    msgs = result.all_messages()
    assert len(msgs) >= 2
    assert isinstance(msgs[0], ModelRequest)
    assert isinstance(msgs[1], ModelResponse)


def test_all_messages_has_user_prompt_part():
    """The first ModelRequest contains a UserPromptPart with the user's question."""
    with explorer_agent.override(model=_model_text_only()):
        result = explorer_agent.run_sync("What datasets exist?")

    first_request = result.all_messages()[0]
    user_parts = [p for p in first_request.parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 1
    assert "What datasets exist?" in user_parts[0].content


def test_all_messages_includes_tool_call_and_return():
    """all_messages() includes ToolCallPart and ToolReturnPart."""
    with explorer_agent.override(model=_model_list_then_answer()):
        result = explorer_agent.run_sync("List datasets")

    msgs = result.all_messages()

    tool_calls = [p for msg in msgs for p in msg.parts if isinstance(p, ToolCallPart)]
    tool_returns = [
        p for msg in msgs for p in msg.parts if isinstance(p, ToolReturnPart)
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "list_datasets"
    assert len(tool_returns) == 1
    assert "sales" in str(tool_returns[0].content)


# --- Test: message_history continues conversation ---


def test_message_history_continues_conversation():
    """Passing message_history includes prior messages in the new run."""
    # Turn 1
    with explorer_agent.override(model=_model_list_then_answer()):
        result1 = explorer_agent.run_sync("What datasets are available?")

    history = result1.all_messages()
    turn1_count = len(history)

    # Turn 2 — uses history from turn 1
    with explorer_agent.override(
        model=_model_text_only("April had the highest revenue.")
    ):
        result2 = explorer_agent.run_sync(
            "Which month had highest revenue?",
            message_history=history,
        )

    msgs2 = result2.all_messages()
    # Turn 2 messages should include turn 1 history + new request/response
    assert len(msgs2) > turn1_count
    # The original user prompt from turn 1 should still be in the history
    all_user_parts = [
        p for msg in msgs2 for p in msg.parts if isinstance(p, UserPromptPart)
    ]
    prompts = [p.content for p in all_user_parts]
    assert "What datasets are available?" in prompts
    assert "Which month had highest revenue?" in prompts


# --- Test: capture_run_messages vs all_messages ---


def test_capture_run_messages_includes_full_history():
    """capture_run_messages() includes passed-in history.

    Key insight: capture_run_messages() and all_messages() return the same data.
    Both include the message_history that was passed in. There is no built-in way
    to isolate "just this run's messages" — you'd need to compare lengths or use
    the run_id on each message to filter.
    """
    # Turn 1
    with explorer_agent.override(model=_model_text_only("Datasets listed.")):
        result1 = explorer_agent.run_sync("List datasets")

    history = result1.all_messages()
    turn1_count = len(history)

    # Turn 2 with capture
    with explorer_agent.override(model=_model_text_only("Here is the follow-up.")):
        with capture_run_messages() as captured:
            result2 = explorer_agent.run_sync("Tell me more", message_history=history)

    # capture_run_messages includes the full history + new messages
    assert len(captured) == turn1_count + 2  # history + new request + new response
    # It matches all_messages()
    assert len(captured) == len(result2.all_messages())

    # To isolate "just this run", filter by run_id or slice by position
    new_messages = captured[turn1_count:]
    assert len(new_messages) == 2
    user_parts = [
        p for msg in new_messages for p in msg.parts if isinstance(p, UserPromptPart)
    ]
    assert any("Tell me more" in p.content for p in user_parts)


# --- Test: usage tracking ---


def test_usage_returns_run_usage():
    """result.usage() returns a RunUsage object with token fields."""
    with explorer_agent.override(model=_model_text_only()):
        result = explorer_agent.run_sync("Hello")

    usage = result.usage()
    # FunctionModel doesn't report real tokens, but the object should exist
    assert hasattr(usage, "input_tokens")
    assert hasattr(usage, "output_tokens")
    assert hasattr(usage, "total_tokens")


# --- Test: trim_history ---


def test_trim_history_keeps_last_n():
    """trim_history returns at most keep_last messages."""
    with explorer_agent.override(model=_model_list_then_answer()):
        result = explorer_agent.run_sync("List datasets")

    msgs = result.all_messages()
    # Request, Response(tool_call), Request(tool_return), Response(text)
    assert len(msgs) >= 4

    trimmed = trim_history(msgs, keep_last=2)
    assert len(trimmed) <= 3  # 2 + possibly 1 prepended for tool pair


def test_trim_history_preserves_tool_pairs():
    """If the cut point splits a tool call/return pair, the tool call is prepended."""
    with explorer_agent.override(model=_model_list_then_answer()):
        result = explorer_agent.run_sync("List datasets")

    msgs = result.all_messages()

    # Trim to last 2 — ToolReturnPart needs its preceding ModelResponse
    trimmed = trim_history(msgs, keep_last=2)

    # Check: if first message is a ModelRequest with ToolReturnPart,
    # the preceding ModelResponse with ToolCallPart must also be present
    tool_returns_in_trimmed = [
        p for msg in trimmed for p in msg.parts if isinstance(p, ToolReturnPart)
    ]
    tool_calls_in_trimmed = [
        p for msg in trimmed for p in msg.parts if isinstance(p, ToolCallPart)
    ]
    # If there are tool returns, there must be matching tool calls
    if tool_returns_in_trimmed:
        assert len(tool_calls_in_trimmed) >= len(tool_returns_in_trimmed)


def test_trim_history_noop_when_short():
    """trim_history returns the full list when it's already within the limit."""
    with explorer_agent.override(model=_model_text_only()):
        result = explorer_agent.run_sync("Hello")

    msgs = result.all_messages()
    trimmed = trim_history(msgs, keep_last=10)
    assert len(trimmed) == len(msgs)


# --- Test: multi-turn with tool then follow-up ---


def test_multi_turn_tool_then_followup():
    """Full multi-turn flow: tool call in turn 1, text follow-up in turn 2."""
    # Turn 1: agent calls query_dataset
    call_count = 0

    def turn1_handler(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_dataset",
                        args={"dataset_name": "sales"},
                    )
                ]
            )
        return ModelResponse(
            parts=[TextPart(content="Sales data has 4 months of records.")]
        )

    with explorer_agent.override(model=FunctionModel(turn1_handler)):
        result1 = explorer_agent.run_sync("Show me sales data")

    history = result1.all_messages()

    # Turn 2: follow-up referencing turn 1 context (no tool call needed)
    with explorer_agent.override(
        model=_model_text_only("April had the highest revenue at 21000.")
    ):
        result2 = explorer_agent.run_sync(
            "Which month had the highest revenue?",
            message_history=history,
        )

    assert "21000" in result2.output
    # Turn 2 history should contain both turns
    msgs2 = result2.all_messages()
    user_prompts = [
        p.content for msg in msgs2 for p in msg.parts if isinstance(p, UserPromptPart)
    ]
    assert "Show me sales data" in user_prompts
    assert "Which month had the highest revenue?" in user_prompts
