"""The :class:`Tool` data type.

In v0.1.0, tools are constructed directly. The ``@tool`` decorator lands in
the next step and produces these instances from typed Python functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A callable the model can invoke during a :func:`step`.

    ``parameters_schema`` is a JSON Schema object describing ``fn``'s
    arguments — what the model receives. ``raise_on_error`` opts out of the
    default model-recovery behavior; when ``True``, exceptions from ``fn``
    propagate as :class:`bridle.errors.ToolExecutionError`.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., Any]
    raise_on_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


__all__ = ["Tool"]
