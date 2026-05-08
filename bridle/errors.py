"""Bridle exception hierarchy.

All Bridle-raised errors descend from :class:`BridleError`. Catching it gives
you a single seam to handle anything the library throws.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import ValidationError


class BridleError(Exception):
    """Base for all Bridle errors."""


class SchemaSatisfactionError(BridleError):
    """The model could not produce output matching the requested schema."""

    def __init__(
        self,
        message: str,
        *,
        schema: type[Any] | None = None,
        last_attempt: Any = None,
        validation_error: ValidationError | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.schema = schema
        self.last_attempt = last_attempt
        self.validation_error = validation_error
        self.attempts = attempts


class ToolExecutionError(BridleError):
    """A tool raised and the user opted out of model recovery."""

    def __init__(
        self, message: str, *, tool_name: str | None = None, cause: BaseException | None = None
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.cause = cause


class ModelError(BridleError):
    """The provider failed after retries are exhausted."""


class LoopExhaustedError(BridleError):
    """A :func:`loop` hit its iteration cap before its predicate returned True."""

    def __init__(self, message: str, *, iterations: int = 0, accumulator: Any = None) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.accumulator = accumulator


class TimeoutError(BridleError):
    """A wrapped call exceeded its deadline."""


class ConfigurationError(BridleError):
    """A required piece of configuration was missing or invalid."""


class TokenBudgetExceededError(BridleError):
    """An agent's token budget was exhausted mid-run."""

    def __init__(self, message: str, *, used: int = 0, budget: int = 0) -> None:
        super().__init__(message)
        self.used = used
        self.budget = budget


__all__ = [
    "BridleError",
    "ConfigurationError",
    "LoopExhaustedError",
    "ModelError",
    "SchemaSatisfactionError",
    "TimeoutError",
    "TokenBudgetExceededError",
    "ToolExecutionError",
]
