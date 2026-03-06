"""Chapter 5: Message History & Conversational State.

Demonstrates:
- result.all_messages() to capture the full message exchange
- Passing message_history to continue a conversation across runs
- Message types: ModelRequest, ModelResponse, UserPromptPart, TextPart,
  ToolCallPart, ToolReturnPart
- capture_run_messages() to inspect a single run's messages
- result.usage() to track token consumption per turn
- Sliding window history trimming (keeping tool call pairs intact)
"""

from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.usage import RunUsage

# --- Model Setup (Pattern A) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Mock Dataset ---
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

# --- Agent ---
explorer_agent = Agent(
    default_model,
    instructions=(
        "You are a data exploration assistant. "
        "Use the available tools to look up datasets and answer questions. "
        "When the user asks follow-up questions, use context from the conversation."
    ),
)


@explorer_agent.tool_plain
def list_datasets() -> str:
    """List all available datasets and their column names."""
    lines = []
    for name, rows in MOCK_DATASETS.items():
        cols = ", ".join(rows[0].keys()) if rows else "(empty)"
        lines.append(f"- {name}: columns [{cols}], {len(rows)} rows")
    return "\n".join(lines)


@explorer_agent.tool_plain
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


# --- Sliding Window History Trimming ---


def trim_history(
    messages: list[ModelMessage], keep_last: int = 4
) -> list[ModelMessage]:
    """Trim message history to the last N messages, preserving tool call pairs.

    A ModelResponse with ToolCallParts must be followed by its matching
    ModelRequest with ToolReturnParts — splitting them would cause API errors.

    Args:
        messages: Full message history from all_messages().
        keep_last: Number of messages to keep from the end.
    """
    if len(messages) <= keep_last:
        return list(messages)

    trimmed = list(messages[-keep_last:])

    # If the first message is a ModelRequest containing ToolReturnParts,
    # we need the preceding ModelResponse (the tool call) too.
    first = trimmed[0]
    if isinstance(first, ModelRequest):
        has_tool_return = any(isinstance(p, ToolReturnPart) for p in first.parts)
        if has_tool_return:
            # Find the preceding ModelResponse in the original list
            cut_index = len(messages) - keep_last
            if cut_index > 0 and isinstance(messages[cut_index - 1], ModelResponse):
                trimmed.insert(0, messages[cut_index - 1])

    return trimmed


# --- Convenience runners ---


def explore(
    question: str, history: list[ModelMessage] | None = None
) -> tuple[str, list[ModelMessage], RunUsage]:
    """Explore data with optional conversation history.

    Returns (answer, messages, usage).
    """
    result = explorer_agent.run_sync(question, message_history=history)
    return result.output, result.all_messages(), result.usage()


if __name__ == "__main__":
    # Turn 1
    print("=== Turn 1 ===")
    answer1, messages1, usage1 = explore("What datasets are available?")
    print(f"Answer: {answer1}")
    print(f"Usage: in={usage1.input_tokens}, out={usage1.output_tokens}")
    print(f"Messages in history: {len(messages1)}")

    # Turn 2 — follow-up using history from turn 1
    print("\n=== Turn 2 (with history) ===")
    answer2, messages2, usage2 = explore("Show me the sales data", messages1)
    print(f"Answer: {answer2}")
    print(f"Usage: in={usage2.input_tokens}, out={usage2.output_tokens}")
    print(f"Messages in history: {len(messages2)}")
    print(f"  -> input_tokens grew: {usage1.input_tokens} -> {usage2.input_tokens}")

    # Turn 3 — demonstrate trimming
    print("\n=== Turn 3 (trimmed history) ===")
    trimmed = trim_history(messages2, keep_last=4)
    print(f"History before trim: {len(messages2)} messages")
    print(f"History after trim:  {len(trimmed)} messages")
    answer3, messages3, usage3 = explore(
        "Which month had the highest revenue?", trimmed
    )
    print(f"Answer: {answer3}")
    print(f"Usage: input_tokens={usage3.input_tokens}")

    # Inspect message structure
    print("\n=== Message Structure (Turn 1) ===")
    for i, msg in enumerate(messages1):
        print(f"  [{i}] {type(msg).__name__}")
        for part in msg.parts:
            print(f"       {type(part).__name__}: {str(part)[:80]}")
