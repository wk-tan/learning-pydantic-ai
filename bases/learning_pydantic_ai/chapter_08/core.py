"""Chapter 8: Observability with Pydantic Logfire.

Demonstrates:
- logfire.configure() to set up tracing
- logfire.instrument_pydantic_ai() for automatic agent tracing
- Per-agent instrumentation via Agent(..., instrument=True)
- Custom spans with logfire.span() for application-level tracing
- Viewing the full ReAct loop (LLM → tool call → tool return) in traces
"""

from __future__ import annotations

from typing import Literal

import logfire
from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Logfire Setup ---
# configure() connects to your Logfire project.
# instrument_pydantic_ai() patches all agents to emit OpenTelemetry spans.
logfire.configure()
logfire.instrument_pydantic_ai()

# --- Model Setup ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Data (reused from Chapter 3) ---
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

# --- Agent ---
data_agent = Agent(
    default_model,
    instructions=(
        "You are a data analyst assistant. "
        "Use the provided tools to query tables and calculate aggregations. "
        "Always query the data first, then use calculate_aggregation "
        "for any numeric computations."
    ),
)


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
    """Ask the data agent a question, wrapped in a custom Logfire span."""
    with logfire.span("ask_data_question", question=question):
        result = data_agent.run_sync(question)
        logfire.info(
            "Agent completed",
            output=result.output,
            total_tokens=result.usage().total_tokens,
        )
        return result.output


if __name__ == "__main__":
    # Run 1: Multi-tool query — observe the ReAct trace in Logfire
    print("=== Multi-tool Query ===")
    answer = ask_data_question("What is the average order amount?")
    print(answer)

    # Run 2: Trigger a ModelRetry — observe retry spans in the trace
    print("\n=== Query with Retry ===")
    answer = ask_data_question("How many products are out of stock (stock = 0)?")
    print(answer)
