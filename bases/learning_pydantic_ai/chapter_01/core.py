"""Chapter 1: Agent Fundamentals.

Demonstrates:
- Creating an Agent with a model string and static instructions
- Running agents with run_sync(), run() (async), and run_stream()
- Accessing AgentRunResult.output
- Model-agnostic design: swapping models without changing agent logic
"""

import asyncio

from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Model Setup ---
# Pattern A: pass API key programmatically from pydantic-settings.
# No os.environ mutation needed — the provider receives the key directly.
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- Agent Definition ---
# A single agent can be reused across sync, async, and streaming calls.
# Static instructions tell the LLM its role and constraints.
data_engineering_agent = Agent(
    default_model,
    instructions=(
        "You are a data engineering expert. "
        "Answer questions about data pipelines, ETL/ELT patterns, "
        "data warehousing, and data quality. "
        "Keep answers concise — three sentences max."
    ),
)


def run_sync_example(question: str) -> str:
    """Run the agent synchronously and return the text output."""
    result = data_engineering_agent.run_sync(question)
    return result.output


async def run_async_example(question: str) -> str:
    """Run the agent asynchronously and return the text output."""
    result = await data_engineering_agent.run(question)
    return result.output


async def run_stream_example(question: str) -> str:
    """Run the agent with streaming and return the full text output."""
    async with data_engineering_agent.run_stream(question) as stream:
        # Collects all streamed chunks and returns the final assembled string.
        # Use `async for chunk in stream.stream_text()` to process chunk by chunk.
        result = await stream.get_output()
    return result


def run_with_different_model(
    question: str,
    model: AnthropicModel | str,
) -> str:
    """Demonstrate model-agnostic design by overriding the model at runtime."""
    result = data_engineering_agent.run_sync(question, model=model)
    return result.output


if __name__ == "__main__":
    sample_question = "What is the difference between ETL and ELT?"

    # 1. Synchronous run
    print("=== Sync Run ===")
    sync_output = run_sync_example(sample_question)
    print(sync_output)

    # 2. Async run
    print("\n=== Async Run ===")
    async_output = asyncio.run(run_async_example(sample_question))
    print(async_output)

    # 3. Streaming run
    print("\n=== Stream Run ===")
    stream_output = asyncio.run(run_stream_example(sample_question))
    print(stream_output)

    # 4. Model swap — same agent logic, different model
    # Reuses the same provider (API key) but swaps the model name.
    print("\n=== Model Swap (Haiku) ===")
    haiku_model = AnthropicModel(
        "claude-haiku-4-5",
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    )
    swapped_output = run_with_different_model(sample_question, model=haiku_model)
    print(swapped_output)
