"""Tests for Chapter 3: Function Tools.

Demonstrates:
- TestModel auto-calls registered tools
- FunctionModel simulates multi-step tool call sequences
- capture_run_messages() to inspect the tool call chain
- ModelRetry behavior when tool receives invalid args
"""

from learning_pydantic_ai.chapter_03.core import (
    calculate_aggregation,
    data_agent,
    query_table,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Tool unit tests (no LLM needed) ---


def test_query_table_returns_rows():
    """query_table returns stringified rows for a valid table."""
    result = query_table("orders")
    assert "150.0" in result
    assert "completed" in result


def test_query_table_raises_model_retry_for_unknown_table():
    """ModelRetry is raised for unknown table names."""
    try:
        query_table("nonexistent")
        assert False, "Should have raised ModelRetry"
    except Exception as e:
        assert "not found" in str(e)
        assert "orders" in str(e)  # Suggests available tables


def test_calculate_aggregation_avg():
    result = calculate_aggregation([100.0, 200.0, 300.0], "avg")
    assert result == "200.0"


def test_calculate_aggregation_sum():
    result = calculate_aggregation([10.0, 20.0, 30.0], "sum")
    assert result == "60.0"


def test_calculate_aggregation_empty_raises_model_retry():
    try:
        calculate_aggregation([], "sum")
        assert False, "Should have raised ModelRetry"
    except Exception as e:
        assert "empty" in str(e)


# --- TestModel: verifies tools are registered and called ---


def test_agent_has_tools_registered():
    """Verify the agent has both tools registered via FunctionModel.

    Note: TestModel generates dummy args (e.g. table_name='a') which trigger
    ModelRetry and exhaust retries. FunctionModel gives us control over args.
    """
    call_count = 0

    def call_one_tool(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "orders"},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done.")])

    with data_agent.override(model=FunctionModel(call_one_tool)):
        with capture_run_messages() as messages:
            result = data_agent.run_sync("Show me orders")
            assert isinstance(result.output, str)

    tool_calls = [
        part.tool_name
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert "query_table" in tool_calls


# --- FunctionModel: simulate the multi-step ReAct loop ---


def test_function_model_multi_step_tool_calls():
    """Simulate: query_table → calculate_aggregation → final answer.

    The FunctionModel steps through the sequence based on call count.
    """
    call_count = 0

    def multi_step(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # Step 1: LLM calls query_table
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "orders"},
                    )
                ]
            )
        elif call_count == 2:
            # Step 2: LLM calls calculate_aggregation with extracted amounts
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="calculate_aggregation",
                        args={
                            "numbers": [150.0, 200.0, 75.0, 320.0, 50.0],
                            "operation": "avg",
                        },
                    )
                ]
            )
        else:
            # Step 3: LLM produces final text answer
            return ModelResponse(
                parts=[TextPart(content="The average order amount is $159.00")]
            )

    with data_agent.override(model=FunctionModel(multi_step)):
        result = data_agent.run_sync("What is the average order amount?")
        assert "159" in result.output
        assert call_count == 3


def test_model_retry_triggers_self_correction():
    """Simulate: LLM tries invalid table → gets ModelRetry → tries correct table."""
    call_count = 0

    def retry_sequence(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # LLM tries an invalid table name
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "sales"},
                    )
                ]
            )
        elif call_count == 2:
            # After receiving ModelRetry feedback, LLM corrects itself
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "orders"},
                    )
                ]
            )
        else:
            return ModelResponse(parts=[TextPart(content="Found 5 orders.")])

    with data_agent.override(model=FunctionModel(retry_sequence)):
        with capture_run_messages() as messages:
            result = data_agent.run_sync("Show me the sales data")
            assert "5" in result.output

    # Verify the retry happened: ModelRetry sends a RetryPromptPart back
    retry_parts = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, RetryPromptPart)
    ]
    assert len(retry_parts) > 0, "Should have a RetryPromptPart from ModelRetry"
    assert "not found" in str(retry_parts[0].content)


def test_capture_run_messages_shows_tool_chain():
    """capture_run_messages() captures the full tool call sequence."""
    call_count = 0

    def simple_sequence(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "products"},
                    )
                ]
            )
        else:
            return ModelResponse(parts=[TextPart(content="There are 3 products.")])

    with data_agent.override(model=FunctionModel(simple_sequence)):
        with capture_run_messages() as messages:
            data_agent.run_sync("How many products are there?")

    # Messages should contain: user prompt, tool call, tool return, final text
    assert len(messages) >= 3
