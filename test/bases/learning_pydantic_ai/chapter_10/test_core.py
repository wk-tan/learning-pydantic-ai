"""Tests for Chapter 10: MCP Integration.

Demonstrates:
- Unit testing MCP server tools directly (no subprocess needed)
- Integration testing the agent with a real MCP server subprocess
- TestModel with MCP toolsets for tool discovery verification
- FunctionModel to simulate specific tool call sequences over MCP
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic_ai import capture_run_messages, models
from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False

# --- MCP server tool unit tests (no subprocess needed) ---


def test_list_tables_returns_available_tables():
    """Server tool list_tables returns both table names."""
    from learning_pydantic_ai.chapter_10.server import list_tables

    result = list_tables()
    assert "orders" in result
    assert "products" in result


def test_query_table_returns_rows():
    """Server tool query_table returns row data for a valid table."""
    from learning_pydantic_ai.chapter_10.server import query_table

    result = query_table("orders")
    assert "150.0" in result
    assert "completed" in result


def test_query_table_returns_error_for_unknown_table():
    """Server tool query_table returns an error message for unknown tables."""
    from learning_pydantic_ai.chapter_10.server import query_table

    result = query_table("nonexistent")
    assert "not found" in result
    assert "orders" in result  # suggests available tables


# --- Integration tests: agent + MCP server subprocess ---

SERVER_SCRIPT = str(
    Path(__file__).parents[4]
    / "bases"
    / "learning_pydantic_ai"
    / "chapter_10"
    / "server.py"
)


def _make_mcp_agent(model: TestModel | FunctionModel) -> Agent:
    """Create a fresh agent with MCP toolset for testing."""
    server = MCPServerStdio(
        "uv",
        args=["run", "python", SERVER_SCRIPT],
        timeout=30,
    )
    return Agent(
        model,
        instructions=(
            "You are a data analyst. Use list_tables to discover tables, "
            "then query_table to get data. Give concise answers."
        ),
        toolsets=[server],
    )


@pytest.mark.anyio
async def test_agent_discovers_mcp_tools_with_test_model():
    """TestModel can discover and call MCP tools via stdio subprocess."""
    agent = _make_mcp_agent(TestModel())

    with capture_run_messages() as messages:
        result = await agent.run("What tables are available?")

    assert isinstance(result.output, str)

    # TestModel should have attempted to call the discovered MCP tools
    tool_calls = [
        part.tool_name
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert len(tool_calls) > 0, "TestModel should call at least one MCP tool"


@pytest.mark.anyio
async def test_agent_calls_mcp_tools_with_function_model():
    """FunctionModel drives specific tool call sequence over MCP."""
    call_count = 0

    def multi_step(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_tables", args={})])
        elif call_count == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": "orders"},
                    )
                ]
            )
        else:
            return ModelResponse(
                parts=[TextPart(content="Total order amount is $795.00")]
            )

    agent = _make_mcp_agent(FunctionModel(multi_step))

    with capture_run_messages() as messages:
        result = await agent.run("What is the total amount across all orders?")

    assert "795" in result.output
    assert call_count == 3

    # Verify tool call sequence in messages
    tool_calls = [
        part.tool_name
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert tool_calls == ["list_tables", "query_table"]
