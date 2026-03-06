"""Chapter 6: Streaming.

Demonstrates:
- agent.run_stream() returning StreamedRunResult
- stream_text() for token-by-token text output (delta=True vs delta=False)
- Streaming structured output with incremental validation
- How streaming interacts with tool calls (tools resolve first, then answer streams)
- result.all_messages() works the same on streamed results
"""

import asyncio

from learning_pydantic_ai.settings.core import settings
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Model Setup (Pattern A) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Dataset (reused from Chapter 5) ---
MOCK_DATASETS: dict[str, list[dict]] = {
    "sales": [
        {"month": "Jan", "revenue": 12000, "orders": 150},
        {"month": "Feb", "revenue": 15000, "orders": 180},
        {"month": "Mar", "revenue": 9000, "orders": 110},
        {"month": "Apr", "revenue": 21000, "orders": 250},
    ],
    "products": [
        {"name": "Widget A", "price": 29.99, "stock": 500},
        {"name": "Widget B", "price": 49.99, "stock": 200},
        {"name": "Widget C", "price": 9.99, "stock": 1000},
    ],
}


# --- Structured Output Model ---
class DataSummary(BaseModel):
    """Structured summary of a dataset analysis."""

    dataset_name: str
    row_count: int
    key_insight: str


# === Section 1: Basic Text Streaming ===

text_agent = Agent(
    default_model,
    instructions="You are a helpful data assistant. Keep answers concise.",
)


async def stream_text_demo() -> None:
    """Stream text token-by-token using stream_text(delta=True)."""
    print("=== Basic Text Streaming (delta=True) ===")
    async with text_agent.run_stream(
        "Explain what streaming is in 2 sentences."
    ) as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end="", flush=True)
    print("\n")

    # Compare with delta=False (cumulative)
    print("=== Cumulative Text Streaming (delta=False) ===")
    async with text_agent.run_stream("What is ETL? One sentence.") as result:
        async for text_so_far in result.stream_text(delta=False):
            print(f"\r{text_so_far}", end="", flush=True)
    print("\n")


# === Section 2: Structured Output Streaming ===

structured_agent = Agent(
    default_model,
    output_type=DataSummary,
    instructions=(
        "You are a data analyst. Analyze the given dataset "
        "and return a structured summary."
    ),
)


async def stream_structured_demo() -> None:
    """Stream structured output — partial objects build up, final result is validated.

    NOTE: stream_text() raises UserError on structured output (tool-based mode).
    Use stream_output() instead — it yields partial Pydantic models as tokens arrive.
    """
    print("=== Structured Output Streaming ===")
    async with structured_agent.run_stream(
        "Summarize the sales dataset: "
        "Jan $12k/150 orders, Feb $15k/180, "
        "Mar $9k/110, Apr $21k/250"
    ) as result:
        async for partial in result.stream_output():
            print(f"\r  partial → {partial}", end="", flush=True)
        print()

        # After consuming the stream, get the final validated output
        output = await result.get_output()
        print(f"\nValidated output: {output}")
        print(f"  dataset_name = {output.dataset_name!r}")
        print(f"  row_count    = {output.row_count}")
        print(f"  key_insight  = {output.key_insight!r}")


# === Section 3: Streaming with Tools ===

tool_agent = Agent(
    default_model,
    instructions=(
        "You are a data exploration assistant. "
        "Use tools to look up data, then stream your answer."
    ),
)


@tool_agent.tool_plain
def list_datasets() -> str:
    """List all available datasets and their column names."""
    lines = []
    for name, rows in MOCK_DATASETS.items():
        cols = ", ".join(rows[0].keys()) if rows else "(empty)"
        lines.append(f"- {name}: columns [{cols}], {len(rows)} rows")
    return "\n".join(lines)


@tool_agent.tool_plain
def query_dataset(dataset_name: str) -> str:
    """Retrieve all rows from a dataset.

    Args:
        dataset_name: Name of the dataset (e.g. 'sales', 'products').
    """
    if dataset_name not in MOCK_DATASETS:
        available = ", ".join(MOCK_DATASETS.keys())
        return f"Dataset '{dataset_name}' not found. Available: {available}"
    rows = MOCK_DATASETS[dataset_name]
    header = " | ".join(rows[0].keys())
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row.values()))
    return "\n".join(lines)


async def stream_with_tools_demo() -> None:
    """Demonstrate that tool calls resolve first, then the final answer streams."""
    print("=== Streaming with Tools ===")
    print("(Tool calls happen first, then the answer streams)\n")
    async with tool_agent.run_stream("Which product has the lowest price?") as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end="", flush=True)
    print()

    # Messages work the same as non-streaming
    print(f"\nMessages captured: {len(result.all_messages())}")
    for i, msg in enumerate(result.all_messages()):
        print(f"  [{i}] {type(msg).__name__}")
        for part in msg.parts:
            print(f"       {type(part).__name__}: {str(part)[:80]}")


# === Main ===

if __name__ == "__main__":

    async def main() -> None:
        await stream_text_demo()
        await stream_structured_demo()
        await stream_with_tools_demo()

    asyncio.run(main())
