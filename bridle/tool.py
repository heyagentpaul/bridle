"""The :class:`Tool` data type and the ``@tool`` decorator.

Tools are typed Python callables the model can invoke during a :func:`step`.
The decorator extracts the parameter schema from type hints (via Pydantic),
the description from the first line of the docstring, and packages the
function into a :class:`Tool`. The resulting :class:`Tool` is itself callable
— ``my_tool(...)`` invokes the underlying function — so unit-testing a tool
is just calling it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints, overload

from pydantic import create_model


@dataclass(frozen=True)
class Tool:
    """A typed callable the model can invoke during a :func:`step`.

    ``parameters_schema`` is a JSON Schema object describing the function's
    arguments. ``raise_on_error`` opts out of the default model-recovery
    behavior; when ``True``, exceptions from ``fn`` propagate as
    :class:`bridle.errors.ToolExecutionError` instead of being fed back to
    the model as recoverable tool results.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    fn: Callable[..., Any]
    raise_on_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped function. Useful for unit-testing tools as plain callables."""

        return self.fn(*args, **kwargs)


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    raise_on_error: bool = False,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    raise_on_error: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function into a :class:`Tool`.

    Use bare or with arguments::

        @tool
        def search(query: str) -> list[str]:
            \"\"\"Web search.\"\"\"
            ...

        @tool(name="lookup", raise_on_error=True)
        def lookup(id: str) -> Record:
            ...
    """

    def decorator(target: Callable[..., Any]) -> Tool:
        return _build_tool(
            target,
            name=name,
            description=description,
            raise_on_error=raise_on_error,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


def _build_tool(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    raise_on_error: bool,
) -> Tool:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation: Any = hints.get(param_name, Any)
        default: Any = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    params_schema: dict[str, Any]
    if fields:
        params_model = create_model(f"{fn.__name__}_Params", **fields)
        params_schema = dict(params_model.model_json_schema())
    else:
        params_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    if description is not None:
        desc = description
    elif fn.__doc__:
        desc = inspect.cleandoc(fn.__doc__).split("\n\n", 1)[0].strip()
    else:
        desc = fn.__name__

    return Tool(
        name=name or fn.__name__,
        description=desc,
        parameters_schema=params_schema,
        fn=fn,
        raise_on_error=raise_on_error,
    )


__all__ = ["Tool", "tool"]
