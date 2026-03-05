"""Chapter 2: Structured Output & Type Safety.

Demonstrates:
- Setting output_type to a Pydantic BaseModel
- Output validation with field validators (triggers LLM self-correction on failure)
- Controlling retry behavior with output_retries
- Union types / multiple output types
- Native vs tool-based output modes (conceptual — mode depends on model support)
"""

from __future__ import annotations

from typing import Literal

from learning_pydantic_ai.settings.core import settings
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

# --- Model Setup (same Pattern A as Chapter 1) ---
default_model = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key=settings.anthropic_api_key),
)


# --- Output Models ---
class DataQualityReport(BaseModel):
    """Structured report for a data quality issue."""

    table: str
    column: str
    issue_type: Literal["null", "type_mismatch", "range_violation"]
    severity: int

    @field_validator("severity")
    @classmethod
    def severity_in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            msg = "severity must be between 1 and 5"
            raise ValueError(msg)
        return v


class DataQualityOK(BaseModel):
    """Returned when the description indicates no data quality issue."""

    summary: str


# List of output types for the triage agent — PydanticAI registers each as a
# separate tool and lets the LLM choose which one to call.
DataQualityOutputTypes = [DataQualityReport, DataQualityOK]


# --- Agents ---

# Agent with a single structured output type.
# PydanticAI generates JSON Schema from DataQualityReport and sends it to the LLM.
# If the LLM returns invalid output (e.g. severity=10), pydantic validation fails
# and PydanticAI automatically asks the LLM to fix its response.
report_agent = Agent(
    default_model,
    output_type=DataQualityReport,
    output_retries=2,
    instructions=(
        "You are a data quality analyst. "
        "Given a natural language description of a data quality issue, "
        "extract the structured fields: table name, column name, issue type, "
        "and severity (1=minor, 5=critical)."
    ),
)

# Agent with union output — can return either a report or an OK result.
# PydanticAI presents both schemas to the LLM and lets it choose.
triage_agent = Agent(
    default_model,
    output_type=DataQualityOutputTypes,  # type: ignore[arg-type]
    output_retries=2,
    instructions=(
        "You are a data quality analyst. "
        "Given a description, determine if there is a data quality issue. "
        "If yes, return a DataQualityReport with table, column, "
        "issue_type, and severity (1-5). "
        "If no issue is found, return DataQualityOK with a brief summary."
    ),
)


def analyze_issue(description: str) -> DataQualityReport:
    """Analyze a data quality issue and return a structured report."""
    result = report_agent.run_sync(description)
    return result.output


def triage_issue(description: str) -> DataQualityReport | DataQualityOK:
    """Triage a description — returns a report if there's an issue, or OK if not."""
    result = triage_agent.run_sync(description)
    return result.output  # type: ignore[return-value]


if __name__ == "__main__":
    # 1. Basic structured output
    print("=== Structured Output ===")
    report = analyze_issue(
        "The users table has a lot of NULL values in the email column. "
        "About 30% of rows are affected."
    )
    print(f"Table:      {report.table}")
    print(f"Column:     {report.column}")
    print(f"Issue type: {report.issue_type}")
    print(f"Severity:   {report.severity}")

    # 2. Union output — issue detected
    print("\n=== Triage (issue) ===")
    result = triage_issue(
        "The orders.amount column contains negative values which shouldn't be possible."
    )
    print(f"Type: {type(result).__name__}")
    print(result)

    # 3. Union output — no issue
    print("\n=== Triage (OK) ===")
    result = triage_issue(
        "All columns in the customers table look clean after the last ETL run."
    )
    print(f"Type: {type(result).__name__}")
    print(result)
