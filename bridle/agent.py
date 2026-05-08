"""The ``@agent`` decorator and its dispatcher.

An agent is a typed callable whose body is a Bridle program: regular Python
that calls primitives. Decorating it with ``@agent`` makes invocation lazy —
calling the function returns a :class:`Call`, and the body runs when that
call is resolved. Inside the body, primitives ride context variables that
carry the agent's model and token budget; nested agents inherit and can
override.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from .call import Call, register, resolve
from .errors import ConfigurationError
from .runtime import (
    push_agent_model,
    push_agent_token_budget,
    push_token_usage,
    reset_agent_model,
    reset_agent_token_budget,
    reset_token_usage,
)
from .schema import is_pydantic_model
from .trace import (
    Event,
    Trace,
    current_event_id,
    current_trace,
    push_event_id,
    reset_active_trace,
    reset_event_id,
    set_active_trace,
)

F = TypeVar("F", bound=Callable[..., Any])


def agent(
    *,
    input: Any = None,
    output: Any = None,
    model: str | None = None,
    token_budget: int | None = None,
    name: str | None = None,
) -> Callable[[F], F]:
    """Mark a function as an agent.

    Each call to the decorated function returns a :class:`Call` of kind
    ``agent``. When that call is resolved (read, awaited, or passed to
    :func:`bridle.resolve`), the body runs with the supplied *model* and
    *token_budget* in scope. Inner primitives pick up these values
    automatically.

    *input* and *output* are optional Pydantic schemas. When set, the first
    positional argument is validated against *input* (coerced from a dict if
    needed), and the body's return value is validated against *output*.
    """

    def decorator(fn: F) -> F:
        agent_label = name or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return Call(
                kind="agent",
                options={
                    "fn": fn,
                    "args": args,
                    "kwargs": kwargs,
                    "input_schema": input,
                    "output_schema": output,
                    "agent_model": model,
                    "token_budget": token_budget,
                    "name": agent_label,
                },
            )

        return cast("F", wrapper)

    return decorator


def _coerce_input(schema: Any, value: Any) -> Any:
    """If *schema* is a Pydantic model, coerce *value* through it."""

    if not is_pydantic_model(schema):
        return value
    if isinstance(value, schema):
        return value
    return cast("type[BaseModel]", schema).model_validate(value)


def _coerce_output(schema: Any, value: Any) -> Any:
    if not is_pydantic_model(schema):
        return value
    if isinstance(value, schema):
        return value
    return cast("type[BaseModel]", schema).model_validate(value)


def _dispatch_agent(call: Call) -> Any:
    options = call.options
    fn = options.get("fn")
    if not callable(fn):
        raise ConfigurationError("Agent call is missing a callable body.")
    args: tuple[Any, ...] = options.get("args", ())
    kwargs: dict[str, Any] = options.get("kwargs", {})
    label: str = options.get("name") or "agent"
    agent_model_value: str | None = options.get("agent_model")
    token_budget: int | None = options.get("token_budget")
    input_schema: Any = options.get("input_schema")
    output_schema: Any = options.get("output_schema")

    if input_schema is not None and args:
        head = _coerce_input(input_schema, args[0])
        args = (head, *args[1:])

    parent_trace = current_trace()
    trace = parent_trace if parent_trace is not None else Trace()
    trace_token = set_active_trace(trace) if parent_trace is None else None

    parent_id = current_event_id()
    start_event = Event.new("call_start", parent_id=parent_id, call_kind="agent", label=label)
    trace.emit(start_event)
    event_token = push_event_id(start_event.id)

    model_token = push_agent_model(agent_model_value) if agent_model_value is not None else None
    budget_token = push_agent_token_budget(token_budget) if token_budget is not None else None
    # When the agent declares a budget, reset usage so the budget applies to
    # this agent's subtree only — nested agents within a budgeted agent share
    # the parent's accumulator.
    usage_token = push_token_usage(0) if token_budget is not None else None

    error: BaseException | None = None
    try:
        result = fn(*args, **kwargs)
        result = resolve(result)
        if output_schema is not None:
            result = _coerce_output(output_schema, result)
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace.emit(
            Event.new(
                "call_end",
                call_kind="agent",
                parent_id=start_event.id,
                label=label,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )
        reset_event_id(event_token)
        if usage_token is not None:
            reset_token_usage(usage_token)
        if budget_token is not None:
            reset_agent_token_budget(budget_token)
        if model_token is not None:
            reset_agent_model(model_token)
        if trace_token is not None:
            reset_active_trace(trace_token)


register("agent", _dispatch_agent)


__all__ = ["agent"]
