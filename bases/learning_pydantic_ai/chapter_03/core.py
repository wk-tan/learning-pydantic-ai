"""Chapter 3: Function Tools — Giving Agents Capabilities.

Demonstrates:
- @agent.tool_plain for standalone tools (no RunContext needed)
- Docstrings become tool descriptions sent to the LLM
- Function signatures become the tool's JSON Schema
- The ReAct loop: LLM reasons → calls tool → observes result → reasons again
- ModelRetry for validation within tools (LLM self-corrects on bad args)
"""

from __future__ import annotations

from typing import Literal

from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Model Setup (same Pattern A as Chapter 1) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Data ---
# Simulates BigQuery tables with hardcoded rows.
MOCK_TABLES: dict[str, list[dict]] = {
    "orders": [
        {"id": 1, "amount": 150.0, "status": "completed"},
        {"id": 2, "amount": 200.0, "status": "pending"},
        {"id": 3, "amount": 75.0, "status": "completed"},
        {"id": 4, "amount": 320.0, "status": "completed"},
        {"id": 5, "amount": 50.0, "status": "cancelled"},
    ],
    "products": [
        {"id": 1, "name": "Widget A", "price": 25.0, "stock": 100},
        {"id": 2, "name": "Widget B", "price": 50.0, "stock": 0},
        {"id": 3, "name": "Widget C", "price": 75.0, "stock": 45},
    ],
}


# --- Agent with Tools ---
data_agent = Agent(
    default_model,
    instructions=(
        "You are a data analyst assistant. "
        "Use the provided tools to query tables and calculate aggregations. "
        "Always query the data first, then use calculate_aggregation "
        "for any numeric computations."
    ),
)


# Tool docstrings are sent to the LLM as the tool description.
# Function signatures become the tool's JSON Schema (parameter names, types).
@data_agent.tool_plain(retries=2)
def query_table(table_name: str) -> str:
    """Query a BigQuery table and return its rows.

    Args:
        table_name: Name of the table to query (e.g. 'orders', 'products').
    """
    if table_name not in MOCK_TABLES:
        raise ModelRetry(
            f"Table '{table_name}' not found. "
            f"Available tables: {', '.join(MOCK_TABLES.keys())}"
        )
    return str(MOCK_TABLES[table_name])


@data_agent.tool_plain
def calculate_aggregation(
    numbers: list[float],
    operation: Literal["sum", "avg", "max", "min"],
) -> str:
    """Calculate an aggregation over a list of numbers.

    Args:
        numbers: List of numeric values to aggregate.
        operation: The aggregation to perform — one of 'sum', 'avg', 'max', 'min'.
    """
    if not numbers:
        raise ModelRetry("Cannot aggregate an empty list of numbers.")
    ops = {
        "sum": sum(numbers),
        "avg": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers),
    }
    result = ops[operation]
    return str(result)


def ask_data_question(question: str) -> str:
    """Ask the data agent a question that may require tool calls."""
    result = data_agent.run_sync(question)
    return result.output


if __name__ == "__main__":
    # The agent must chain two tools: query_table → calculate_aggregation
    print("=== Multi-tool Query ===")
    answer = ask_data_question("What is the average order amount?")
    print(answer)

    # ModelRetry demo: the agent might try an invalid table name,
    # get corrected, and then use the right one.
    print("\n=== Query with context ===")
    answer = ask_data_question("How many products are out of stock (stock = 0)?")
    print(answer)
