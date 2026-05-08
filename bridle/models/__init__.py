"""Model client protocol and the value types it exchanges.

A :class:`ModelClient` takes a list of messages and a list of tools, runs
one turn against a provider, and returns a :class:`ModelResponse`. The tool
loop in :mod:`bridle._internal.tool_loop` calls it repeatedly until the
model produces a valid structured return.

In v0.1.0 the only real implementation is the Anthropic adapter (added in
the next step). Tests use :class:`MockModelClient` from
:mod:`bridle.models.mock`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ..tool import Tool

StopReason = Literal["end_turn", "tool_use", "max_tokens", "other"]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model emitted in a turn."""

    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ModelResponse:
    """One turn's worth of model output."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: StopReason = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class ModelClient(Protocol):
    """Provider-neutral interface used by the tool loop."""

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[Tool],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        """Run one turn and return what the model said."""

        ...


__all__ = [
    "ModelClient",
    "ModelResponse",
    "StopReason",
    "ToolCall",
]
