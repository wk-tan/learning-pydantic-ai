"""Chapter 1: Agent Fundamentals.

Demonstrates:
- Creating an Agent with a model string and static instructions
- Running agents with run_sync(), run() (async), and run_stream()
- Accessing AgentRunResult.output
- Model-agnostic design: swapping models without changing agent logic
"""

import asyncio

from pydantic_ai import Agent

# --- Agent Definition ---
# A single agent can be reused across sync, async, and streaming calls.
# Static instructions tell the LLM its role and constraints.
# Using model=None makes the agent testable without API keys at import time.
# The model is provided at runtime via run_sync(..., model=...) or override().
data_engineering_agent = Agent(
    instructions=(
        "You are a data engineering expert. "
        "Answer questions about data pipelines, ETL/ELT patterns, "
        "data warehousing, and data quality. "
        "Keep answers concise — three sentences max."
    ),
)

DEFAULT_MODEL = "anthropic:claude-sonnet-4-6"


def run_sync_example(question: str) -> str:
    """Run the agent synchronously and return the text output."""
    result = data_engineering_agent.run_sync(question, model=DEFAULT_MODEL)
    return result.output


async def run_async_example(question: str) -> str:
    """Run the agent asynchronously and return the text output."""
    result = await data_engineering_agent.run(question, model=DEFAULT_MODEL)
    return result.output


async def run_stream_example(question: str) -> str:
    """Run the agent with streaming and return the full text output."""
    async with data_engineering_agent.run_stream(
        question, model=DEFAULT_MODEL
    ) as stream:
        # Collects all streamed chunks and returns the final assembled string.
        # Use `async for chunk in stream.stream_text()` to process chunk by chunk.
        result = await stream.get_output()
    return result


def run_with_different_model(question: str, model: str) -> str:
    """Demonstrate model-agnostic design by overriding the model at runtime."""
    result = data_engineering_agent.run_sync(question, model=model)
    return result.output


if __name__ == "__main__":
    import os

    from learning_pydantic_ai.settings.core import settings

    # pydantic-settings loads .env into the Settings object, but PydanticAI's
    # providers read from os.environ directly — bridge the gap here.
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

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
    # Requires OPENAI_API_KEY in .env; skipped gracefully if unavailable.
    print("\n=== Model Swap (OpenAI) ===")
    if os.environ.get("OPENAI_API_KEY"):
        swapped_output = run_with_different_model(
            sample_question, model="openai:gpt-4o"
        )
        print(swapped_output)
    else:
        print("Skipped — OPENAI_API_KEY not set")
