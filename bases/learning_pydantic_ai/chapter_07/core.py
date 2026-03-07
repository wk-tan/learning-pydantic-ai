"""Chapter 7: Testing — TestModel, FunctionModel & Override.

Demonstrates PydanticAI's testing toolkit:
- TestModel for quick, procedural tests (no LLM)
- FunctionModel for controlling exact model responses
- Agent.override() to swap models without changing call sites
- models.ALLOW_MODEL_REQUESTS = False as a safety guard
- capture_run_messages() for asserting on tool call sequences
- Overriding deps in tests to inject mock services

The agent under test is a data catalog assistant (adapted from Chapter 4).
"""

from typing import Literal

from learning_pydantic_ai.settings.core import settings
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Model Setup (Pattern A) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Data Catalog ---
FULL_CATALOG: dict[str, dict] = {
    "public.orders": {
        "columns": ["id", "amount", "status", "created_at"],
        "description": "Customer order records",
        "access": "analyst",
    },
    "public.products": {
        "columns": ["id", "name", "price", "stock"],
        "description": "Product inventory",
        "access": "analyst",
    },
    "internal.customers_pii": {
        "columns": ["id", "name", "email", "phone", "address"],
        "description": "Customer PII — restricted",
        "access": "admin",
    },
    "internal.revenue_forecast": {
        "columns": ["quarter", "projected", "actual", "variance"],
        "description": "Revenue forecasting model outputs",
        "access": "admin",
    },
}


# --- Dependencies ---
class CatalogDeps(BaseModel):
    """Runtime dependencies injected into the agent at call time."""

    user_role: Literal["analyst", "admin"]
    catalog: dict[str, dict]


# --- Structured Output ---
class TableSummary(BaseModel):
    """Structured output: summary of a table lookup."""

    table_name: str
    description: str
    column_count: int


# --- Agent ---
catalog_agent = Agent(
    default_model,
    deps_type=CatalogDeps,
    output_type=str,
)


@catalog_agent.instructions
def catalog_instructions(ctx: RunContext[CatalogDeps]) -> str:
    role = ctx.deps.user_role
    if role == "admin":
        return (
            "You are a data catalog assistant with full admin access. "
            "You can see all tables including restricted ones."
        )
    return (
        "You are a data catalog assistant. "
        "You have analyst-level access and can only see public tables."
    )


@catalog_agent.tool
def list_tables(ctx: RunContext[CatalogDeps]) -> str:
    """List all tables the current user has access to.

    Returns a summary of each visible table with its columns and description.
    """
    role = ctx.deps.user_role
    catalog = ctx.deps.catalog
    visible = {
        name: info
        for name, info in catalog.items()
        if role == "admin" or info["access"] == "analyst"
    }
    if not visible:
        return "No tables available for your access level."
    lines = []
    for name, info in visible.items():
        cols = ", ".join(info["columns"])
        lines.append(f"- {name}: {info['description']} (columns: {cols})")
    return "\n".join(lines)


@catalog_agent.tool
def get_table_details(ctx: RunContext[CatalogDeps], table_name: str) -> str:
    """Get detailed information about a specific table.

    Args:
        table_name: Fully qualified table name (e.g. 'public.orders').
    """
    catalog = ctx.deps.catalog
    if table_name not in catalog:
        raise ModelRetry(
            f"Table '{table_name}' not found. Use list_tables to see available tables."
        )
    info = catalog[table_name]
    role = ctx.deps.user_role
    if role != "admin" and info["access"] != "analyst":
        raise ModelRetry(
            f"Access denied: '{table_name}' requires admin access. "
            f"Use list_tables to see tables you can access."
        )
    cols = ", ".join(info["columns"])
    return f"Table: {table_name}\nDescription: {info['description']}\nColumns: {cols}"


# --- Second agent for structured output testing ---
summary_agent = Agent(
    default_model,
    deps_type=CatalogDeps,
    output_type=TableSummary,
)


@summary_agent.tool
def lookup_table(ctx: RunContext[CatalogDeps], table_name: str) -> str:
    """Look up a table and return its details for summarization.

    Args:
        table_name: Fully qualified table name (e.g. 'public.orders').
    """
    catalog = ctx.deps.catalog
    if table_name not in catalog:
        raise ModelRetry(f"Table '{table_name}' not found in catalog.")
    info = catalog[table_name]
    cols = ", ".join(info["columns"])
    return (
        f"Table: {table_name}\n"
        f"Description: {info['description']}\n"
        f"Columns: {cols}\n"
        f"Column count: {len(info['columns'])}"
    )


if __name__ == "__main__":
    deps = CatalogDeps(user_role="analyst", catalog=FULL_CATALOG)
    result = catalog_agent.run_sync("What tables can I access?", deps=deps)
    print(result.output)
