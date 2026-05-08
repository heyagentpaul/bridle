"""Error hierarchy: every Bridle error descends from ``BridleError``."""

from __future__ import annotations

import pytest

from bridle.errors import (
    BridleError,
    ConfigurationError,
    LoopExhaustedError,
    ModelError,
    SchemaSatisfactionError,
    TimeoutError,
    TokenBudgetExceededError,
    ToolExecutionError,
)


@pytest.mark.parametrize(
    "cls",
    [
        ConfigurationError,
        LoopExhaustedError,
        ModelError,
        SchemaSatisfactionError,
        TimeoutError,
        TokenBudgetExceededError,
        ToolExecutionError,
    ],
)
def test_descends_from_bridle_error(cls: type[BridleError]) -> None:
    assert issubclass(cls, BridleError)


def test_schema_satisfaction_carries_metadata() -> None:
    err = SchemaSatisfactionError("nope", schema=int, last_attempt={"x": 1}, attempts=3)
    assert err.schema is int
    assert err.last_attempt == {"x": 1}
    assert err.attempts == 3


def test_loop_exhausted_carries_accumulator() -> None:
    err = LoopExhaustedError("hit cap", iterations=32, accumulator=[1, 2, 3])
    assert err.iterations == 32
    assert err.accumulator == [1, 2, 3]


def test_tool_execution_carries_cause() -> None:
    cause = ValueError("boom")
    err = ToolExecutionError("tool failed", tool_name="search", cause=cause)
    assert err.tool_name == "search"
    assert err.cause is cause


def test_token_budget_carries_usage() -> None:
    err = TokenBudgetExceededError("over", used=12_000, budget=10_000)
    assert err.used == 12_000
    assert err.budget == 10_000
