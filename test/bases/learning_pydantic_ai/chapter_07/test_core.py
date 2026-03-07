"""Tests for Chapter 7: Testing — TestModel, FunctionModel & Override.

Demonstrates ALL testing techniques in one place:

1. ALLOW_MODEL_REQUESTS = False — safety guard against real API calls
2. TestModel — procedural output, auto-calls tools, fast
3. FunctionModel — exact control over model responses
4. Agent.override() — swap model without changing agent code
5. capture_run_messages() — assert on tool call sequences
6. Deps override — inject mock services in tests
7. TestModel custom_output_args — satisfy custom validators
"""

from typing import Literal

from learning_pydantic_ai.chapter_07.core import (
    FULL_CATALOG,
    CatalogDeps,
    TableSummary,
    catalog_agent,
    summary_agent,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

# ============================================================
# 1. Safety Guard — ALLOW_MODEL_REQUESTS = False
# ============================================================
# This is the FIRST line in a test module. It ensures that if
# any test forgets to override(), a real API call will raise
# an error instead of silently spending money.
models.ALLOW_MODEL_REQUESTS = False


# --- Shared helpers ---


def _deps(role: Literal["analyst", "admin"] = "analyst") -> CatalogDeps:
    return CatalogDeps(user_role=role, catalog=FULL_CATALOG)


def _small_catalog_deps(role: Literal["analyst", "admin"] = "analyst") -> CatalogDeps:
    """A minimal catalog for testing deps override."""
    return CatalogDeps(
        user_role=role,
        catalog={
            "test.only_table": {
                "columns": ["id", "value"],
                "description": "Test-only table",
                "access": "analyst",
            },
        },
    )


def _extract_tool_returns(messages: list, tool_name: str | None = None) -> list[str]:
    """Extract tool return content strings from captured messages."""
    results = []
    for msg in messages:
        for part in msg.parts if hasattr(msg, "parts") else []:
            if isinstance(part, ToolReturnPart):
                if tool_name is None or part.tool_name == tool_name:
                    results.append(str(part.content))
    return results


def _extract_retry_prompts(messages: list) -> list[str]:
    """Extract retry prompt content strings from captured messages."""
    results = []
    for msg in messages:
        for part in msg.parts if hasattr(msg, "parts") else []:
            if isinstance(part, RetryPromptPart):
                results.append(str(part.content))
    return results


# ============================================================
# 2. TestModel — Procedural, Zero-Config Tests
# ============================================================
# TestModel auto-generates dummy args and calls all registered
# tools. Great for "smoke tests" that verify wiring.


class TestTestModel:
    """Demonstrate TestModel behavior.

    KEY LESSON: TestModel generates dummy args for tool parameters
    (e.g. 'a' for str, 0 for int). If your tools have domain validation
    (like checking if a table exists), the dummy args will trigger
    ModelRetry and eventually exceed max retries.

    Solutions:
    - Use call_tools='no' to skip tool execution (test output only)
    - Use FunctionModel when you need to test specific tool sequences
    """

    def test_smoke_test_with_no_tool_calls(self):
        """TestModel with call_tools=[] — verifies agent wiring without tools."""
        with catalog_agent.override(model=TestModel(call_tools=[])):
            result = catalog_agent.run_sync("Show me tables", deps=_deps("analyst"))

        # TestModel returns a generic text response
        assert isinstance(result.output, str)

    def test_dummy_args_trigger_model_retry(self):
        """TestModel sends 'a' as table_name → triggers ModelRetry → exceeds retries.

        This is expected! It demonstrates WHY FunctionModel exists.
        """
        import pytest
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        with catalog_agent.override(model=TestModel()):
            with pytest.raises(UnexpectedModelBehavior, match="exceeded max retries"):
                catalog_agent.run_sync("Show me tables", deps=_deps("analyst"))

    def test_testmodel_with_custom_output_args(self):
        """custom_output_args controls structured output from TestModel."""
        with summary_agent.override(
            model=TestModel(
                call_tools=[],
                custom_output_args={
                    "table_name": "public.orders",
                    "description": "Customer orders",
                    "column_count": 4,
                },
            )
        ):
            result = summary_agent.run_sync(
                "Summarize public.orders", deps=_deps("analyst")
            )

        assert isinstance(result.output, TableSummary)
        assert result.output.table_name == "public.orders"
        assert result.output.column_count == 4


# ============================================================
# 3. FunctionModel — Exact Control Over Responses
# ============================================================
# FunctionModel lets you script exactly what the "LLM" returns
# at each step. Essential for testing specific tool sequences.


class TestFunctionModel:
    """Demonstrate FunctionModel for scripted responses."""

    def test_scripted_tool_call_sequence(self):
        """FunctionModel scripts: call list_tables → return text."""
        call_count = 0

        def scripted(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            return ModelResponse(parts=[TextPart(content="You have 2 public tables.")])

        with catalog_agent.override(model=FunctionModel(scripted)):
            result = catalog_agent.run_sync("What tables?", deps=_deps("analyst"))

        assert result.output == "You have 2 public tables."
        assert call_count == 2

    def test_multi_step_tool_sequence(self):
        """FunctionModel can script multi-step flows: list → details → text."""
        call_count = 0

        def multi_step(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            if call_count == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_table_details",
                            args={"table_name": "public.orders"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="Orders has 4 columns.")])

        with catalog_agent.override(model=FunctionModel(multi_step)):
            with capture_run_messages() as msgs:
                result = catalog_agent.run_sync(
                    "Tell me about orders", deps=_deps("analyst")
                )

        assert result.output == "Orders has 4 columns."
        assert call_count == 3

        # Verify both tools were called in sequence
        tool_calls = [
            part.tool_name
            for msg in msgs
            for part in (msg.parts if hasattr(msg, "parts") else [])
            if isinstance(part, ToolCallPart)
        ]
        assert tool_calls == ["list_tables", "get_table_details"]

    def test_model_retry_appears_in_messages(self):
        """When a tool raises ModelRetry, it becomes a RetryPromptPart."""
        call_count = 0

        def try_bad_table(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_table_details",
                            args={"table_name": "nonexistent.table"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="That table doesn't exist.")])

        with catalog_agent.override(model=FunctionModel(try_bad_table)):
            with capture_run_messages() as msgs:
                catalog_agent.run_sync("Show nonexistent.table", deps=_deps("analyst"))

        retries = _extract_retry_prompts(msgs)
        assert len(retries) >= 1
        assert "not found" in retries[0]


# ============================================================
# 4. Agent.override() — Swap Model at Test Time
# ============================================================
# The key pattern: your application code uses the real model,
# but tests swap it out via override(). No code changes needed.


class TestOverride:
    """Demonstrate Agent.override() for model swapping."""

    def test_override_with_test_model(self):
        """override() with TestModel — simplest possible test."""
        with catalog_agent.override(model=TestModel(call_tools=[])):
            result = catalog_agent.run_sync("Hello", deps=_deps("analyst"))
        assert isinstance(result.output, str)

    def test_override_with_function_model(self):
        """override() with FunctionModel — controlled responses."""

        def simple(messages: list, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="Mocked response")])

        with catalog_agent.override(model=FunctionModel(simple)):
            result = catalog_agent.run_sync("Hello", deps=_deps("analyst"))
        assert result.output == "Mocked response"


# ============================================================
# 5. capture_run_messages() — Assert on Tool Call Sequences
# ============================================================


class TestCaptureRunMessages:
    """Demonstrate capture_run_messages() for message inspection."""

    def test_inspect_full_message_chain(self):
        """capture_run_messages() captures the full request/response chain."""
        call_count = 0

        def scripted(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            return ModelResponse(parts=[TextPart(content="Done.")])

        with catalog_agent.override(model=FunctionModel(scripted)):
            with capture_run_messages() as msgs:
                catalog_agent.run_sync("List tables", deps=_deps("analyst"))

        # Messages alternate: ModelRequest → ModelResponse → ModelRequest → ...
        # At minimum: user prompt → tool call → tool return → final text
        assert len(msgs) >= 4

        # Verify tool return contains expected data
        tool_returns = _extract_tool_returns(msgs, "list_tables")
        assert len(tool_returns) == 1
        assert "public.orders" in tool_returns[0]

    def test_analyst_never_sees_restricted_data(self):
        """Security assertion: restricted table data never leaks to analyst."""
        call_count = 0

        def try_restricted(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_table_details",
                            args={"table_name": "internal.customers_pii"},
                        )
                    ]
                )
            if call_count == 2:
                # After retry, LLM falls back to allowed table
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_table_details",
                            args={"table_name": "public.orders"},
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart(content="Here are the orders details.")]
            )

        with catalog_agent.override(model=FunctionModel(try_restricted)):
            with capture_run_messages() as msgs:
                catalog_agent.run_sync(
                    "Show me internal.customers_pii", deps=_deps("analyst")
                )

        # PII columns must never appear in tool returns
        all_returns = _extract_tool_returns(msgs)
        for content in all_returns:
            assert "email" not in content, "PII leaked to analyst"
            assert "phone" not in content, "PII leaked to analyst"


# ============================================================
# 6. Deps Override — Inject Mock Services
# ============================================================
# Same agent, different deps = different behavior.
# In real apps, deps might contain DB clients, HTTP clients, etc.
# In tests, you inject mocks.


class TestDepsOverride:
    """Demonstrate overriding deps to inject mock services."""

    def test_small_catalog_limits_results(self):
        """Injecting a smaller catalog limits what the agent can see."""
        call_count = 0

        def call_list(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            return ModelResponse(parts=[TextPart(content="One table available.")])

        with catalog_agent.override(model=FunctionModel(call_list)):
            with capture_run_messages() as msgs:
                catalog_agent.run_sync("What tables?", deps=_small_catalog_deps())

        tool_returns = _extract_tool_returns(msgs, "list_tables")
        assert len(tool_returns) == 1
        assert "test.only_table" in tool_returns[0]
        assert "public.orders" not in tool_returns[0]

    def test_empty_catalog(self):
        """An empty catalog returns the 'no tables' message."""
        call_count = 0

        def call_list(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            return ModelResponse(parts=[TextPart(content="No tables found.")])

        empty_deps = CatalogDeps(user_role="analyst", catalog={})
        with catalog_agent.override(model=FunctionModel(call_list)):
            with capture_run_messages() as msgs:
                catalog_agent.run_sync("What tables?", deps=empty_deps)

        tool_returns = _extract_tool_returns(msgs, "list_tables")
        assert "No tables available" in tool_returns[0]

    def test_same_agent_different_roles(self):
        """Same agent + same catalog + different role = different visibility."""
        call_count = 0

        def call_list(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name="list_tables", args={})]
                )
            return ModelResponse(parts=[TextPart(content="Done.")])

        # Analyst
        with catalog_agent.override(model=FunctionModel(call_list)):
            with capture_run_messages() as analyst_msgs:
                catalog_agent.run_sync("List tables", deps=_deps("analyst"))

        call_count = 0  # Reset for second run

        # Admin
        with catalog_agent.override(model=FunctionModel(call_list)):
            with capture_run_messages() as admin_msgs:
                catalog_agent.run_sync("List tables", deps=_deps("admin"))

        analyst_content = _extract_tool_returns(analyst_msgs, "list_tables")[0]
        admin_content = _extract_tool_returns(admin_msgs, "list_tables")[0]

        assert "internal." not in analyst_content
        assert "internal.customers_pii" in admin_content
        assert "internal.revenue_forecast" in admin_content


# ============================================================
# 7. Structured Output with TestModel — custom_output_args
# ============================================================


class TestStructuredOutput:
    """Demonstrate testing structured output agents."""

    def test_testmodel_with_custom_output_args(self):
        """custom_output_args lets you control TestModel's structured output."""
        with summary_agent.override(
            model=TestModel(
                call_tools=[],
                custom_output_args={
                    "table_name": "public.products",
                    "description": "Product inventory",
                    "column_count": 4,
                },
            )
        ):
            result = summary_agent.run_sync("Summarize products", deps=_deps("analyst"))

        assert result.output.table_name == "public.products"
        assert result.output.column_count == 4

    def test_function_model_structured_output(self):
        """FunctionModel with structured output via tool call."""
        call_count = 0

        def scripted(messages: list, info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="lookup_table",
                            args={"table_name": "public.orders"},
                        )
                    ]
                )
            # Return structured output via the final_result tool
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={
                            "table_name": "public.orders",
                            "description": "Customer order records",
                            "column_count": 4,
                        },
                    )
                ]
            )

        with summary_agent.override(model=FunctionModel(scripted)):
            result = summary_agent.run_sync("Summarize orders", deps=_deps("analyst"))

        assert isinstance(result.output, TableSummary)
        assert result.output.table_name == "public.orders"
        assert result.output.column_count == 4
