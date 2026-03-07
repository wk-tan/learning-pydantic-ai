"""Tests for Chapter 8: Observability with Pydantic Logfire.

Demonstrates:
- Logfire instrumentation does not break agent behavior
- TestModel + override still works with instrumented agents
- logfire.suppress_instrumentation() disables tracing in tests
"""

import logfire
from learning_pydantic_ai.chapter_08.core import (
    calculate_aggregation,
    data_agent,
    query_table,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Tool unit tests (same as Chapter 3, verifies tools still work) ---


def test_query_table_returns_rows():
    result = query_table("orders")
    assert "150.0" in result


def test_query_table_raises_model_retry_for_unknown_table():
    try:
        query_table("nonexistent")
        assert False, "Should have raised ModelRetry"
    except Exception as e:
        assert "not found" in str(e)


def test_calculate_aggregation_avg():
    result = calculate_aggregation([100.0, 200.0, 300.0], "avg")
    assert result == "200.0"


# --- Agent tests: instrumentation does not interfere with TestModel ---


def test_agent_runs_with_instrumentation_and_function_model():
    """Agent with Logfire instrumentation works normally under FunctionModel."""
    call_count = 0

    def multi_step(messages: list, info: AgentInfo) -> ModelResponse:
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
        elif call_count == 2:
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
            return ModelResponse(
                parts=[TextPart(content="The average order amount is $159.00")]
            )

    with data_agent.override(model=FunctionModel(multi_step)):
        result = data_agent.run_sync("What is the average order amount?")
        assert "159" in result.output
        assert call_count == 3


def test_capture_run_messages_works_with_instrumentation():
    """capture_run_messages() captures messages even with Logfire active."""
    call_count = 0

    def simple(messages: list, info: AgentInfo) -> ModelResponse:
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
        return ModelResponse(parts=[TextPart(content="3 products found.")])

    with data_agent.override(model=FunctionModel(simple)):
        with capture_run_messages() as messages:
            data_agent.run_sync("How many products?")

    assert len(messages) >= 3


def test_suppress_instrumentation():
    """logfire.suppress_instrumentation() silences tracing when needed."""
    call_count = 0

    def simple(messages: list, info: AgentInfo) -> ModelResponse:
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

    with logfire.suppress_instrumentation():
        with data_agent.override(model=FunctionModel(simple)):
            result = data_agent.run_sync("Show orders")
            assert isinstance(result.output, str)
