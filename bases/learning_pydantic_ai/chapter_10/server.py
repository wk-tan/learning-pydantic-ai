"""MCP Server: Mock data catalog exposed via FastMCP.

This server runs as a subprocess (stdio transport) and exposes
tools for listing and querying mock database tables.

Logfire is configured here to emit server-side spans as a separate trace.
"""

from __future__ import annotations

import logfire
from fastmcp import FastMCP

# --- Logfire (server-side trace) ---
logfire.configure(service_name="mcp-data-server")

# --- FastMCP Server ---
mcp = FastMCP("data-catalog")

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


@mcp.tool()
def list_tables() -> str:
    """List all available tables in the data catalog."""
    with logfire.span("list_tables"):
        tables = list(MOCK_TABLES.keys())
        logfire.info("Listed tables", table_count=len(tables))
        return str(tables)


@mcp.tool()
def query_table(table_name: str) -> str:
    """Query a table and return its rows.

    Args:
        table_name: Name of the table to query (e.g. 'orders', 'products').
    """
    with logfire.span("query_table", table_name=table_name):
        if table_name not in MOCK_TABLES:
            available = ", ".join(MOCK_TABLES.keys())
            return f"Error: Table '{table_name}' not found. Available: {available}"
        rows = MOCK_TABLES[table_name]
        logfire.info("Queried table", table_name=table_name, row_count=len(rows))
        return str(rows)


if __name__ == "__main__":
    mcp.run(transport="stdio")
