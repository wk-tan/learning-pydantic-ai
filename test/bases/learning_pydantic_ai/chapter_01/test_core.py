"""Tests for Chapter 1: Agent Fundamentals.

Uses TestModel to avoid real LLM API calls (Chapter 7 preview).
"""

import pytest
from learning_pydantic_ai.chapter_01.core import (
    data_engineering_agent,
    run_async_example,
    run_sync_example,
)
from pydantic_ai import models
from pydantic_ai.models.test import TestModel

# Guard against accidental real API calls in tests
models.ALLOW_MODEL_REQUESTS = False


def test_run_sync_returns_string():
    with data_engineering_agent.override(model=TestModel()):
        result = run_sync_example("What is ETL?")
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.anyio
async def test_run_async_returns_string():
    with data_engineering_agent.override(model=TestModel()):
        result = await run_async_example("What is a data warehouse?")
        assert isinstance(result, str)
        assert len(result) > 0
