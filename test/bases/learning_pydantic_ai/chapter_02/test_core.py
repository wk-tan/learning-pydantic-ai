"""Tests for Chapter 2: Structured Output & Type Safety.

Demonstrates:
- TestModel auto-generates valid output matching the Pydantic schema
- FunctionModel simulates specific LLM responses (including invalid ones to test retry)
- Union output type resolution
"""

from learning_pydantic_ai.chapter_02.core import (
    DataQualityOK,
    DataQualityReport,
    analyze_issue,
    report_agent,
    triage_agent,
    triage_issue,
)
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Basic structured output with TestModel ---


def test_analyze_issue_returns_report():
    """TestModel with custom_output_args to satisfy the severity validator."""
    with report_agent.override(
        model=TestModel(
            custom_output_args={
                "table": "users",
                "column": "email",
                "issue_type": "null",
                "severity": 3,
            }
        )
    ):
        result = analyze_issue("NULL emails in users table")
        assert isinstance(result, DataQualityReport)
        assert result.severity == 3


def test_report_fields_match_custom_args():
    """custom_output_args values flow through to the output."""
    with report_agent.override(
        model=TestModel(
            custom_output_args={
                "table": "orders",
                "column": "amount",
                "issue_type": "type_mismatch",
                "severity": 5,
            }
        )
    ):
        result = analyze_issue("Type mismatch in orders.amount")
        assert result.table == "orders"
        assert result.column == "amount"
        assert result.issue_type == "type_mismatch"
        assert result.severity == 5


# --- FunctionModel: simulate specific LLM responses ---


def test_function_model_returns_specific_report():
    """FunctionModel lets us control the exact LLM response."""

    def return_specific_report(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "table": "orders",
                        "column": "amount",
                        "issue_type": "range_violation",
                        "severity": 4,
                    },
                )
            ]
        )

    with report_agent.override(model=FunctionModel(return_specific_report)):
        result = analyze_issue("Negative amounts in orders table")
        assert result.table == "orders"
        assert result.column == "amount"
        assert result.issue_type == "range_violation"
        assert result.severity == 4


def test_validation_triggers_retry():
    """When the LLM returns invalid output, PydanticAI retries.

    First call: severity=10 (invalid, must be 1-5).
    Second call: severity=3 (valid).
    """
    call_count = 0

    def invalid_then_valid(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        severity = 10 if call_count == 1 else 3
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "table": "events",
                        "column": "timestamp",
                        "issue_type": "type_mismatch",
                        "severity": severity,
                    },
                )
            ]
        )

    with report_agent.override(model=FunctionModel(invalid_then_valid)):
        result = analyze_issue("Timestamp column has string values")
        assert result.severity == 3
        assert call_count == 2  # Confirms retry happened


# --- Union output types ---


def test_triage_returns_report_via_test_model():
    """TestModel with custom_output_args matching DataQualityReport."""
    with triage_agent.override(
        model=TestModel(
            custom_output_args={
                "table": "users",
                "column": "email",
                "issue_type": "null",
                "severity": 1,
            }
        )
    ):
        result = triage_issue("Some description")
        assert isinstance(result, (DataQualityReport, DataQualityOK))


def test_triage_returns_report_via_function_model():
    """Force the triage agent to return a DataQualityReport."""

    def return_report(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result_DataQualityReport",
                    args={
                        "table": "users",
                        "column": "email",
                        "issue_type": "null",
                        "severity": 2,
                    },
                )
            ]
        )

    with triage_agent.override(model=FunctionModel(return_report)):
        result = triage_issue("Null emails in users table")
        assert isinstance(result, DataQualityReport)
        assert result.table == "users"


def test_triage_returns_ok_via_function_model():
    """Force the triage agent to return DataQualityOK."""

    def return_ok(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result_DataQualityOK",
                    args={"summary": "All data looks clean"},
                )
            ]
        )

    with triage_agent.override(model=FunctionModel(return_ok)):
        result = triage_issue("Everything looks good")
        assert isinstance(result, DataQualityOK)
        assert result.summary == "All data looks clean"
