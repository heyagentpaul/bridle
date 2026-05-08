"""Mock model client for tests and the ``mock`` wrapper.

Construct with either a list of pre-baked :class:`ModelResponse`\\ s (popped
in order) or a callable that produces a response from each request. Both
are deterministic — no I/O, no real provider — so unit tests can assert on
exact behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..tool import Tool
from . import ModelResponse, ToolCall

ResponseFactory = Callable[
    [Sequence[dict[str, Any]], Sequence[Tool]],
    ModelResponse,
]


class MockModelClient:
    """A scripted :class:`bridle.models.ModelClient` implementation."""

    def __init__(self, responses: Sequence[ModelResponse] | ResponseFactory) -> None:
        self._factory: ResponseFactory | None
        if callable(responses):
            self._factory = responses
            self._queue: list[ModelResponse] = []
        else:
            self._factory = None
            self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[Tool],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": list(tools),
                "system": system,
                "params": params,
            }
        )
        if self._factory is not None:
            return self._factory(messages, tools)
        if not self._queue:
            raise RuntimeError(
                "MockModelClient exhausted — scripted responses ran out before the step terminated."
            )
        return self._queue.pop(0)


def text_response(text: str) -> ModelResponse:
    """Helper: a turn that returns text only, no tool calls."""

    return ModelResponse(text=text, tool_calls=(), stop_reason="end_turn")


def tool_response(*calls: ToolCall) -> ModelResponse:
    """Helper: a turn that emits one or more tool calls."""

    return ModelResponse(text=None, tool_calls=tuple(calls), stop_reason="tool_use")


__all__ = [
    "MockModelClient",
    "ResponseFactory",
    "text_response",
    "tool_response",
]
