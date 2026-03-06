"""Tests for Chapter 4: Dependency Injection & RunContext.

Demonstrates:
- Passing deps= to control tool behavior
- Admin sees all tables, analyst sees only public ones
- Dynamic instructions change based on deps
- ModelRetry blocks access to restricted tables for analysts
- capture_run_messages() verifies restricted data never leaks
"""

from learning_pydantic_ai.chapter_04.core import (
    FULL_CATALOG,
    CatalogDeps,
    catalog_agent,
)
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Helper to build a RunContext-like deps for direct tool testing ---


def _make_deps(role: str = "analyst") -> CatalogDeps:
    return CatalogDeps(user_role=role, catalog=FULL_CATALOG)


# --- Direct tool unit tests (no LLM) ---


def test_list_tables_analyst_sees_only_public():
    """Analyst should only see tables with access='analyst'."""
    call_count = 0

    def call_list(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_tables", args={})])
        return ModelResponse(parts=[TextPart(content="Here are your tables.")])

    deps = _make_deps("analyst")
    with catalog_agent.override(model=FunctionModel(call_list)):
        with capture_run_messages() as messages:
            catalog_agent.run_sync("What tables can I see?", deps=deps)

    # Find the tool return for list_tables
    tool_returns = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, ToolReturnPart) and part.tool_name == "list_tables"
    ]
    assert len(tool_returns) == 1
    content = str(tool_returns[0].content)
    assert "public.orders" in content
    assert "public.products" in content
    assert "internal.customers_pii" not in content
    assert "internal.revenue_forecast" not in content


def test_list_tables_admin_sees_all():
    """Admin should see all tables including restricted ones."""
    call_count = 0

    def call_list(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_tables", args={})])
        return ModelResponse(parts=[TextPart(content="Here are all tables.")])

    deps = _make_deps("admin")
    with catalog_agent.override(model=FunctionModel(call_list)):
        with capture_run_messages() as messages:
            catalog_agent.run_sync("What tables can I see?", deps=deps)

    tool_returns = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, ToolReturnPart) and part.tool_name == "list_tables"
    ]
    assert len(tool_returns) == 1
    content = str(tool_returns[0].content)
    assert "public.orders" in content
    assert "internal.customers_pii" in content
    assert "internal.revenue_forecast" in content


def test_get_table_details_analyst_blocked_from_admin_table():
    """Analyst requesting an admin-only table triggers ModelRetry."""
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
        elif call_count == 2:
            # After denial, LLM falls back to an allowed table
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_table_details",
                        args={"table_name": "public.orders"},
                    )
                ]
            )
        return ModelResponse(
            parts=[TextPart(content="Here are the orders table details.")]
        )

    deps = _make_deps("analyst")
    with catalog_agent.override(model=FunctionModel(try_restricted)):
        with capture_run_messages() as messages:
            catalog_agent.run_sync("Show me internal.customers_pii", deps=deps)

    # Verify restricted table data never appears in any tool return
    all_tool_returns = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, ToolReturnPart)
    ]
    for tr in all_tool_returns:
        content = str(tr.content)
        assert "email" not in content, "PII columns must not leak to analyst"
        assert "phone" not in content, "PII columns must not leak to analyst"


def test_get_table_details_admin_can_access_restricted():
    """Admin can access restricted tables without ModelRetry."""
    call_count = 0

    def get_restricted(messages: list, info: AgentInfo) -> ModelResponse:
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
        return ModelResponse(
            parts=[TextPart(content="Here are the customer PII details.")]
        )

    deps = _make_deps("admin")
    with catalog_agent.override(model=FunctionModel(get_restricted)):
        with capture_run_messages() as messages:
            catalog_agent.run_sync("Show me internal.customers_pii", deps=deps)

    tool_returns = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, ToolReturnPart) and part.tool_name == "get_table_details"
    ]
    assert len(tool_returns) == 1
    content = str(tool_returns[0].content)
    assert "email" in content
    assert "internal.customers_pii" in content


def test_nonexistent_table_triggers_model_retry():
    """Requesting a table that doesn't exist triggers ModelRetry."""
    call_count = 0

    def try_nonexistent(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="get_table_details",
                        args={"table_name": "public.nonexistent"},
                    )
                ]
            )
        elif call_count == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="list_tables",
                        args={},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                TextPart(
                    content="That table doesn't exist. Here are the available ones."
                )
            ]
        )

    deps = _make_deps("analyst")
    with catalog_agent.override(model=FunctionModel(try_nonexistent)):
        with capture_run_messages() as messages:
            catalog_agent.run_sync("Show me public.nonexistent", deps=deps)

    # Should have a retry prompt with "not found"
    from pydantic_ai.messages import RetryPromptPart

    retry_parts = [
        part
        for msg in messages
        for part in (msg.parts if hasattr(msg, "parts") else [])
        if isinstance(part, RetryPromptPart)
    ]
    assert len(retry_parts) > 0
    assert "not found" in str(retry_parts[0].content)


def test_same_agent_different_deps_different_results():
    """The same agent produces different results based on deps — the key insight."""
    call_count_analyst = 0
    call_count_admin = 0

    def analyst_call(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count_analyst
        call_count_analyst += 1
        if call_count_analyst == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_tables", args={})])
        return ModelResponse(parts=[TextPart(content="2 tables visible.")])

    def admin_call(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count_admin
        call_count_admin += 1
        if call_count_admin == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_tables", args={})])
        return ModelResponse(parts=[TextPart(content="4 tables visible.")])

    # Analyst run
    with catalog_agent.override(model=FunctionModel(analyst_call)):
        with capture_run_messages() as analyst_msgs:
            catalog_agent.run_sync("List tables", deps=_make_deps("analyst"))

    # Admin run
    with catalog_agent.override(model=FunctionModel(admin_call)):
        with capture_run_messages() as admin_msgs:
            catalog_agent.run_sync("List tables", deps=_make_deps("admin"))

    # Extract tool returns
    def get_list_tables_content(msgs):
        for msg in msgs:
            for part in msg.parts if hasattr(msg, "parts") else []:
                if isinstance(part, ToolReturnPart) and part.tool_name == "list_tables":
                    return str(part.content)
        return ""

    analyst_content = get_list_tables_content(analyst_msgs)
    admin_content = get_list_tables_content(admin_msgs)

    # Analyst sees 2 tables, admin sees 4
    assert analyst_content.count("public.") == 2
    assert "internal." not in analyst_content
    assert admin_content.count("public.") == 2
    assert admin_content.count("internal.") == 2
