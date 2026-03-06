"""Chapter 4: Dependency Injection & RunContext.

Demonstrates:
- Defining a deps_type as a BaseModel
- RunContext[YourDepsType] as the typed gateway to dependencies
- Passing deps= at runtime via agent.run()
- Dynamic instructions via @agent.instructions that access ctx.deps
- How deps scope data access (authorization boundary outside the LLM's reach)
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
# Full catalog with access-level metadata per table.
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


# --- Agent ---
catalog_agent = Agent(
    default_model,
    deps_type=CatalogDeps,
)


# Dynamic instructions — the system prompt adapts based on who is calling.
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


# @agent.tool (not tool_plain) — first param is RunContext.
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


def ask_catalog(
    question: str, user_role: Literal["analyst", "admin"] = "analyst"
) -> str:
    """Ask the catalog agent a question with a given role."""
    deps = CatalogDeps(user_role=user_role, catalog=FULL_CATALOG)
    result = catalog_agent.run_sync(question, deps=deps)
    return result.output


if __name__ == "__main__":
    print("=== Analyst View ===")
    answer = ask_catalog("What tables can I access?", user_role="analyst")
    print(answer)

    print("\n=== Admin View ===")
    answer = ask_catalog("What tables can I access?", user_role="admin")
    print(answer)

    print("\n=== Analyst tries restricted table ===")
    answer = ask_catalog(
        "Show me details of internal.customers_pii", user_role="analyst"
    )
    print(answer)
