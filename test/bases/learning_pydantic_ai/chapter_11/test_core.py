"""Tests for Chapter 11: Multi-Agent Patterns.

Demonstrates:
- TestModel with agent.override() for both router and specialist agents
- Verifying delegation flow: router calls the correct specialist via tools
- Shared usage tracking via ctx.usage across delegated runs
- Hand-off pattern: app code routes to agents sequentially
"""

import pytest
from learning_pydantic_ai.chapter_11.core import (
    RouterDeps,
    data_quality_agent,
    router_agent,
    schema_design_agent,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# === Test: Router delegates to specialist agents via tools ===


@pytest.mark.anyio
async def test_router_delegates_via_tool():
    """Router agent should call a delegation tool, producing tool call/return pairs."""
    with (
        router_agent.override(model=TestModel()),
        data_quality_agent.override(model=TestModel()),
        schema_design_agent.override(model=TestModel()),
    ):
        with capture_run_messages() as msgs:
            result = await router_agent.run(
                "Check for null values in the users table.",
                deps=RouterDeps(),
            )

    assert isinstance(result.output, str)
    assert len(result.output) > 0

    # Verify tool calls happened in the message flow
    tool_calls = [p for msg in msgs for p in msg.parts if isinstance(p, ToolCallPart)]
    tool_returns = [
        p for msg in msgs for p in msg.parts if isinstance(p, ToolReturnPart)
    ]
    assert len(tool_calls) >= 1
    assert len(tool_returns) >= 1

    # TestModel calls all tools — both delegation tools should appear
    tool_names = {tc.tool_name for tc in tool_calls}
    assert "delegate_to_data_quality" in tool_names
    assert "delegate_to_schema_design" in tool_names


@pytest.mark.anyio
async def test_router_message_structure():
    """Router messages: Request -> Response(tools) -> Request(returns) -> Response."""
    with (
        router_agent.override(model=TestModel()),
        data_quality_agent.override(model=TestModel()),
        schema_design_agent.override(model=TestModel()),
    ):
        result = await router_agent.run(
            "Design a schema for events.",
            deps=RouterDeps(),
        )

    msgs = result.all_messages()

    # At minimum: user request, tool call response, tool return request, final response
    assert len(msgs) >= 4
    assert isinstance(msgs[0], ModelRequest)
    assert isinstance(msgs[1], ModelResponse)
    assert isinstance(msgs[-1], ModelResponse)


# === Test: Shared usage tracking ===


@pytest.mark.anyio
async def test_shared_usage_tracking():
    """Usage passed via ctx.usage aggregates tokens from router + specialists."""
    usage = RunUsage()

    with (
        router_agent.override(model=TestModel()),
        data_quality_agent.override(model=TestModel()),
        schema_design_agent.override(model=TestModel()),
    ):
        await router_agent.run(
            "Check for type mismatches.",
            deps=RouterDeps(),
            usage=usage,
        )

    # TestModel reports some usage — the key assertion is that usage is non-zero
    # and includes contributions from both router and specialist
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    assert usage.total_tokens is not None
    assert usage.total_tokens > 0


@pytest.mark.anyio
async def test_usage_accumulates_across_runs():
    """Multiple runs with the same RunUsage object accumulate tokens."""
    usage = RunUsage()

    with (
        router_agent.override(model=TestModel()),
        data_quality_agent.override(model=TestModel()),
        schema_design_agent.override(model=TestModel()),
    ):
        await router_agent.run(
            "Check nulls.",
            deps=RouterDeps(),
            usage=usage,
        )
        tokens_after_first = usage.total_tokens

        await router_agent.run(
            "Design a table.",
            deps=RouterDeps(),
            usage=usage,
        )
        tokens_after_second = usage.total_tokens

    assert tokens_after_second > tokens_after_first


# === Test: Hand-off pattern ===


@pytest.mark.anyio
async def test_handoff_pattern():
    """App code routes to different agents sequentially, sharing usage."""
    usage = RunUsage()

    with (
        router_agent.override(model=TestModel()),
        data_quality_agent.override(model=TestModel()),
        schema_design_agent.override(model=TestModel()),
    ):
        # Step 1: Ask router
        result1 = await router_agent.run(
            "Are there nulls in email?",
            deps=RouterDeps(),
            usage=usage,
        )

        # Step 2: Hand off directly to schema agent (app-level routing)
        result2 = await schema_design_agent.run(
            "Add NOT NULL constraint to email.",
            usage=usage,
        )

    assert isinstance(result1.output, str)
    assert isinstance(result2.output, str)
    # Usage should reflect both runs
    assert usage.total_tokens > 0


# === Test: Specialist agents work independently ===


@pytest.mark.anyio
async def test_data_quality_agent_standalone():
    """Data quality agent works as a standalone agent."""
    with data_quality_agent.override(model=TestModel()):
        result = await data_quality_agent.run("Check for null values.")

    assert isinstance(result.output, str)
    assert len(result.output) > 0


@pytest.mark.anyio
async def test_schema_design_agent_standalone():
    """Schema design agent works as a standalone agent."""
    with schema_design_agent.override(model=TestModel()):
        result = await schema_design_agent.run("Design a users table.")

    assert isinstance(result.output, str)
    assert len(result.output) > 0
