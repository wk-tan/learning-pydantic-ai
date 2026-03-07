"""Tests for Chapter 9: Evals — Systematically Measuring Agent Quality.

Demonstrates:
- Building and validating Dataset/Case structure
- Custom evaluator ContainsNumber logic
- Built-in evaluators (EqualsExpected, Contains, IsInstance) behavior
- Running evals with TestModel (no real API calls)
- EvaluationReport inspection
"""

import pytest
from learning_pydantic_ai.chapter_09.core import (
    ContainsNumber,
    build_dataset,
    data_agent,
    run_agent_task,
)
from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected, IsInstance

# Guard against accidental real API calls
models.ALLOW_MODEL_REQUESTS = False


# --- Dataset structure tests ---


def test_build_dataset_has_10_cases():
    dataset = build_dataset()
    assert len(dataset.cases) == 10


def test_each_case_has_name_and_inputs():
    dataset = build_dataset()
    for case in dataset.cases:
        assert case.name is not None
        assert isinstance(case.inputs, str)
        assert len(case.inputs) > 0


def test_dataset_has_isinstance_evaluator():
    dataset = build_dataset()
    assert any(isinstance(e, IsInstance) for e in dataset.evaluators)


# --- Custom evaluator tests ---


def test_contains_number_passes_when_present():
    from pydantic_evals.evaluators import EvaluatorContext

    evaluator = ContainsNumber(expected_number="795")
    ctx = EvaluatorContext(
        name="test",
        inputs="question",
        output="The total is 795.0",
        expected_output="795",
        metadata=None,
        _span_tree=None,  # type: ignore[arg-type]
        duration=0.0,
        attributes={},
        metrics={},
    )
    assert evaluator.evaluate(ctx) is True


def test_contains_number_fails_when_absent():
    from pydantic_evals.evaluators import EvaluatorContext

    evaluator = ContainsNumber(expected_number="795")
    ctx = EvaluatorContext(
        name="test",
        inputs="question",
        output="The total is 500",
        expected_output="795",
        metadata=None,
        _span_tree=None,  # type: ignore[arg-type]
        duration=0.0,
        attributes={},
        metrics={},
    )
    assert evaluator.evaluate(ctx) is False


# --- Agent eval with FunctionModel ---


@pytest.mark.anyio
async def test_eval_with_function_model():
    """Run the eval dataset against a FunctionModel that returns known answers."""
    call_count = 0

    def deterministic_agent(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        # Extract the user prompt from messages
        prompt = ""
        for msg in messages:
            for part in msg.parts:
                if hasattr(part, "content") and isinstance(part.content, str):
                    prompt = part.content

        # Always query_table first, then give a text answer
        if call_count % 2 == 1:
            # Odd calls: figure out which table to query
            table = "orders" if "order" in prompt.lower() else "products"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="query_table",
                        args={"table_name": table},
                    )
                ]
            )
        # Even calls: return a text response with the expected number
        return ModelResponse(parts=[TextPart(content="The answer is 795")])

    # Small dataset for testing
    dataset = Dataset[str, str, None](
        cases=[
            Case(
                name="total_orders",
                inputs="What is the total amount across all orders?",
                expected_output="795",
                evaluators=(ContainsNumber(expected_number="795"),),
            ),
        ],
        evaluators=[IsInstance(type_name="str")],
    )

    with data_agent.override(model=FunctionModel(deterministic_agent)):
        report = await dataset.evaluate(run_agent_task, name="test-eval")

    assert len(report.cases) == 1
    case_result = report.cases[0]
    assert case_result.output == "The answer is 795"


@pytest.mark.anyio
async def test_eval_report_captures_all_scores():
    """Verify the report contains assertions from all evaluators."""

    def simple_agent(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="Widget C costs 75.0")])

    dataset = Dataset[str, str, None](
        cases=[
            Case(
                name="expensive_product",
                inputs="What is the most expensive product?",
                expected_output="Widget C",
                evaluators=(
                    Contains(value="Widget C"),
                    ContainsNumber(expected_number="75"),
                ),
            ),
        ],
        evaluators=[IsInstance(type_name="str")],
    )

    with data_agent.override(model=FunctionModel(simple_agent)):
        report = await dataset.evaluate(run_agent_task, name="test-scores")

    case_result = report.cases[0]
    # The report stores evaluator results in assertions (bool evaluators)
    assert "Contains" in case_result.assertions
    assert "ContainsNumber" in case_result.assertions
    assert "IsInstance" in case_result.assertions
    # All should pass for this case
    assert case_result.assertions["Contains"].value is True
    assert case_result.assertions["ContainsNumber"].value is True
    assert case_result.assertions["IsInstance"].value is True


@pytest.mark.anyio
async def test_equals_expected_evaluator():
    """EqualsExpected passes only on exact match."""

    def exact_agent(messages: list, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="42")])

    dataset = Dataset[str, str, None](
        cases=[
            Case(
                name="exact_match",
                inputs="What is the answer?",
                expected_output="42",
            ),
            Case(
                name="no_match",
                inputs="What is the answer?",
                expected_output="99",
            ),
        ],
        evaluators=[EqualsExpected()],
    )

    with data_agent.override(model=FunctionModel(exact_agent)):
        report = await dataset.evaluate(run_agent_task, name="test-exact")

    # First case: exact match → pass
    assert report.cases[0].assertions["EqualsExpected"].value is True
    # Second case: mismatch → fail
    assert report.cases[1].assertions["EqualsExpected"].value is False
