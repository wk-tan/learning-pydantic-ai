"""Chapter 9: Evals — Systematically Measuring Agent Quality.

Demonstrates:
- pydantic_evals Dataset and Case for defining eval scenarios
- Built-in evaluators: EqualsExpected, Contains, IsInstance
- LLMJudge for subjective quality scoring
- HasMatchingSpan for asserting on tool usage via OTel traces
- Custom evaluators extending the Evaluator base class
- Running experiments and printing reports
- Comparing results across two different models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Contains,
    Evaluator,
    EvaluatorContext,
    HasMatchingSpan,
    IsInstance,
    LLMJudge,
)

# --- Model Setup ---
sonnet_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)
haiku_model = AnthropicModel(
    "claude-haiku-4-5-20251001",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Data ---
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
    sonnet_model,
    instructions=(
        "You are a data analyst assistant. "
        "Use the provided tools to query tables and calculate aggregations. "
        "Always query the data first, then use calculate_aggregation "
        "for any numeric computations. "
        "Give concise answers with the numeric result."
    ),
    instrument=True,
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
    ops: dict[str, float] = {
        "sum": sum(numbers),
        "avg": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers),
    }
    result = ops[operation]
    return str(result)


# --- Custom Evaluator ---
@dataclass(repr=False)
class ContainsNumber(Evaluator[str, str, None]):
    """Check that the agent output contains a specific number."""

    expected_number: str = ""

    def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
        return self.expected_number in ctx.output


# --- Eval Dataset ---
def build_dataset() -> Dataset[str, str, None]:
    """Build the evaluation dataset with 10 data engineering questions."""
    return Dataset(
        cases=[
            # --- Orders table questions ---
            Case(
                name="total_order_amount",
                inputs="What is the total amount across all orders?",
                expected_output="795",
                evaluators=(
                    ContainsNumber(expected_number="795"),
                    HasMatchingSpan(
                        query={"name_contains": "query_table"},
                        evaluation_name="called_query_table",
                    ),
                ),
            ),
            Case(
                name="average_order_amount",
                inputs="What is the average order amount?",
                expected_output="159",
                evaluators=(
                    ContainsNumber(expected_number="159"),
                    HasMatchingSpan(
                        query={"name_contains": "calculate_aggregation"},
                        evaluation_name="called_calculate_aggregation",
                    ),
                ),
            ),
            Case(
                name="max_order_amount",
                inputs="What is the maximum order amount?",
                expected_output="320",
                evaluators=(ContainsNumber(expected_number="320"),),
            ),
            Case(
                name="min_order_amount",
                inputs="What is the minimum order amount?",
                expected_output="50",
                evaluators=(ContainsNumber(expected_number="50"),),
            ),
            Case(
                name="completed_order_count",
                inputs="How many orders have status 'completed'?",
                expected_output="3",
                evaluators=(ContainsNumber(expected_number="3"),),
            ),
            # --- Products table questions ---
            Case(
                name="out_of_stock_count",
                inputs="How many products are out of stock (stock = 0)?",
                expected_output="1",
                evaluators=(ContainsNumber(expected_number="1"),),
            ),
            Case(
                name="most_expensive_product",
                inputs="What is the most expensive product and its price?",
                expected_output="Widget C",
                evaluators=(
                    Contains(value="Widget C"),
                    ContainsNumber(expected_number="75"),
                ),
            ),
            Case(
                name="total_stock",
                inputs="What is the total stock across all products?",
                expected_output="145",
                evaluators=(ContainsNumber(expected_number="145"),),
            ),
            # --- Cross-table questions ---
            Case(
                name="available_tables",
                inputs="What tables are available to query?",
                expected_output="orders, products",
                evaluators=(
                    Contains(value="orders"),
                    Contains(value="products"),
                ),
            ),
            Case(
                name="cheapest_product_name",
                inputs="What is the cheapest product?",
                expected_output="Widget A",
                evaluators=(Contains(value="Widget A"),),
            ),
        ],
        # Dataset-level evaluators applied to ALL cases
        evaluators=[
            IsInstance(type_name="str"),
        ],
    )


async def run_agent_task(question: str) -> str:
    """Task function for the eval framework — runs the agent on a question."""
    result = await data_agent.run(question)
    return result.output


async def run_eval(model_name: str = "sonnet") -> None:
    """Run the full evaluation with a specified model."""

    if model_name == "haiku":
        model = haiku_model
    else:
        model = sonnet_model

    dataset = build_dataset()

    with data_agent.override(model=model):
        report = await dataset.evaluate(
            run_agent_task,
            name=f"data-agent-{model_name}",
        )
    report.print(include_input=True, include_output=True)


async def run_eval_with_judge() -> None:
    """Run evaluation with LLM-as-a-judge for subjective quality."""
    from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model

    set_default_judge_model(sonnet_model)

    dataset = Dataset[str, str, None](
        cases=[
            Case(
                name="explain_total",
                inputs="What is the total amount across all orders?",
                expected_output="795",
            ),
            Case(
                name="explain_stock",
                inputs="Which products are out of stock and why might that matter?",
                expected_output="Widget B is out of stock",
            ),
        ],
        evaluators=[
            LLMJudge(
                rubric=(
                    "The answer should be factually correct based on the data, "
                    "concise, and include the relevant numeric value. "
                    "Give a passing grade if the key facts are present."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    )

    report = await dataset.evaluate(run_agent_task, name="data-agent-judge")
    report.print(include_input=True, include_output=True)


async def compare_models() -> None:
    """Run the same dataset against Sonnet and Haiku, print both reports."""
    dataset = build_dataset()

    print("=" * 60)
    print("MODEL: Claude Sonnet")
    print("=" * 60)
    with data_agent.override(model=sonnet_model):
        sonnet_report = await dataset.evaluate(run_agent_task, name="sonnet")
    sonnet_report.print(include_input=True, include_output=True)

    print("\n" + "=" * 60)
    print("MODEL: Claude Haiku")
    print("=" * 60)
    with data_agent.override(model=haiku_model):
        haiku_report = await dataset.evaluate(run_agent_task, name="haiku")
    haiku_report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    import asyncio

    print("=== Running Eval (Sonnet) ===\n")
    asyncio.run(run_eval("sonnet"))
