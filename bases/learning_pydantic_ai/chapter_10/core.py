"""Chapter 10: MCP Integration — Connecting External Tools & Data.

Demonstrates:
- Building a FastMCP server that exposes tools via stdio transport
- Connecting a PydanticAI agent to the MCP server with MCPServerStdio
- MCP tools appearing alongside function tools in the agent's toolset
- Logfire instrumentation on both client (agent) and server sides
- Custom spans wrapping the full agent interaction
"""

from __future__ import annotations

from pathlib import Path

import logfire
from learning_pydantic_ai.settings.core import settings
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Logfire Setup (client-side trace) ---
logfire.configure()
logfire.instrument_pydantic_ai()

# --- Model Setup ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)

# --- MCP Server (stdio transport) ---
# The server script lives next to this file.
SERVER_SCRIPT = str(Path(__file__).parent / "server.py")

mcp_server = MCPServerStdio(
    "uv",
    args=["run", "python", SERVER_SCRIPT],
    timeout=30,
)

# --- Agent with MCP toolset ---
data_agent = Agent(
    default_model,
    instructions=(
        "You are a data analyst assistant. "
        "Use the available tools to discover and query tables. "
        "First list the available tables, then query the relevant one. "
        "Give concise answers with the numeric result."
    ),
    toolsets=[mcp_server],
)


async def ask_data_question(question: str) -> str:
    """Ask the data agent a question, wrapped in a custom Logfire span."""
    with logfire.span("ask_data_question", question=question):
        result = await data_agent.run(question)
        logfire.info(
            "Agent completed",
            output=result.output,
            total_tokens=result.usage().total_tokens,
        )
        return result.output


async def main() -> None:
    """Run example queries through the MCP-powered agent."""
    # Run 1: Simple query — agent discovers tables via MCP, then queries
    print("=== Query 1: Total order amount ===")
    answer = await ask_data_question("What is the total amount across all orders?")
    print(answer)

    # Run 2: Cross-table — agent lists tables, picks the right one
    print("\n=== Query 2: Out of stock products ===")
    answer = await ask_data_question("How many products are out of stock (stock = 0)?")
    print(answer)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
