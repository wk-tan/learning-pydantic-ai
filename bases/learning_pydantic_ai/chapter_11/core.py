"""Chapter 11: Multi-Agent Patterns.

Demonstrates:
- Agent delegation: a parent agent calls child agents from within a tool
- Agent hand-off: application code decides which agent runs next
- Shared usage tracking via ctx.usage across delegated runs
- Agents as stateless, global objects
"""

import asyncio
from dataclasses import dataclass

from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.usage import RunUsage

# --- Model Setup (Pattern A) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)


# === Section 1: Specialist Agents ===
# These are stateless, global objects — defined once, reused everywhere.

data_quality_agent = Agent(
    default_model,
    instructions=(
        "You are a data quality specialist. "
        "Analyze data issues such as nulls, type mismatches, "
        "and range violations. Be concise and actionable."
    ),
)

schema_design_agent = Agent(
    default_model,
    instructions=(
        "You are a schema design specialist. "
        "Help with table design, column naming, normalization, "
        "and data types. Be concise and actionable."
    ),
)


# === Section 2: Router Agent with Delegation ===
# The router classifies the request and delegates to the right specialist
# from within a tool — the key multi-agent pattern.


@dataclass
class RouterDeps:
    """Dependencies for the router — currently empty, but extensible."""


router_agent: Agent[RouterDeps, str] = Agent(
    default_model,
    instructions=(
        "You are a data engineering router. "
        "Classify each user request as either a data quality issue "
        "or a schema design question, then delegate to the appropriate "
        "specialist using the provided tools. "
        "Return the specialist's response verbatim."
    ),
    deps_type=RouterDeps,
)


@router_agent.tool
async def delegate_to_data_quality(ctx: RunContext[RouterDeps], prompt: str) -> str:
    """Delegate a data quality question to the data quality specialist.

    Args:
        prompt: The user's data quality question to analyze.
    """
    result = await data_quality_agent.run(prompt, usage=ctx.usage)
    return result.output


@router_agent.tool
async def delegate_to_schema_design(ctx: RunContext[RouterDeps], prompt: str) -> str:
    """Delegate a schema design question to the schema design specialist.

    Args:
        prompt: The user's schema design question to analyze.
    """
    result = await schema_design_agent.run(prompt, usage=ctx.usage)
    return result.output


# === Section 3: Hand-off Pattern ===
# Application code decides which agent runs — no tool delegation needed.


async def handoff_demo() -> None:
    """Demonstrate the hand-off pattern: app code routes to agents sequentially."""
    print("=== Hand-off Pattern ===\n")

    # Shared usage tracker across all agent runs
    usage = RunUsage()

    # Step 1: Ask the router a data quality question
    result1 = await router_agent.run(
        "Are there null values in the users.email column?",
        deps=RouterDeps(),
        usage=usage,
    )
    print(f"Router response:\n  {result1.output}\n")

    # Step 2: Follow up directly with the schema design agent (hand-off)
    result2 = await schema_design_agent.run(
        "Given a users table with email issues, "
        "what constraints should I add to prevent nulls?",
        usage=usage,
    )
    print(f"Schema agent response:\n  {result2.output}\n")

    # Combined usage from both runs
    print("=== Combined Usage ===")
    print(f"  Total input tokens:  {usage.input_tokens}")
    print(f"  Total output tokens: {usage.output_tokens}")
    print(f"  Total tokens:        {usage.total_tokens}")


# === Section 4: Delegation Demo ===


async def delegation_demo() -> None:
    """Demonstrate the delegation pattern: router delegates via tools."""
    print("=== Delegation Pattern ===\n")

    usage = RunUsage()

    # Data quality question → router delegates to data_quality_agent
    result = await router_agent.run(
        "Check if the orders table has any type mismatches "
        "in the amount column — some values look like strings.",
        deps=RouterDeps(),
        usage=usage,
    )
    print(f"Response:\n  {result.output}\n")

    # Schema design question → router delegates to schema_design_agent
    result = await router_agent.run(
        "Design a normalized schema for a product catalog "
        "with categories, tags, and pricing tiers.",
        deps=RouterDeps(),
        usage=usage,
    )
    print(f"Response:\n  {result.output}\n")

    # Usage includes tokens from BOTH the router AND the specialists
    print("=== Combined Usage (Delegation) ===")
    print(f"  Total input tokens:  {usage.input_tokens}")
    print(f"  Total output tokens: {usage.output_tokens}")
    print(f"  Total tokens:        {usage.total_tokens}")

    # Inspect messages to see the tool call flow
    print("\n=== Message Flow (last run) ===")
    for i, msg in enumerate(result.all_messages()):
        print(f"  [{i}] {type(msg).__name__}")
        for part in msg.parts:
            print(f"       {type(part).__name__}: {str(part)[:100]}")


# === Main ===

if __name__ == "__main__":

    async def main() -> None:
        await delegation_demo()
        print("\n" + "=" * 60 + "\n")
        await handoff_demo()

    asyncio.run(main())
