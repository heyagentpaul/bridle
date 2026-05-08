"""The three primitives. Each returns a :class:`Call`.

This module wires up the dispatch table by registering each primitive's
evaluator with the :mod:`bridle.call` runtime at import time. Call
``bridle.step(...)`` to construct a step; the work happens when something
reads the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from .call import Call, register, resolve
from .errors import ConfigurationError, LoopExhaustedError, TokenBudgetExceededError
from .runtime import (
    current_agent_model,
    current_model_client,
    current_token_usage,
    effective_token_budget,
    require_model,
)
from .tool import Tool
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

T = TypeVar("T")
DEFAULT_LOOP_MAX_ITERATIONS = 32


def step(
    prompt: str,
    *,
    schema: type[T],
    context: Any = None,
    tools: Sequence[Tool] = (),
    label: str | None = None,
) -> T:
    """Run one unit of judgment.

    The model is given *prompt* (and optionally *context*), the supplied
    *tools*, and a synthetic ``__bridle_return__`` tool whose schema matches
    *schema*. When the model calls ``__bridle_return__`` with valid arguments,
    the step returns the typed value.

    Returns a :class:`bridle.Call` that resolves to a value of type ``T`` on
    first use. The annotation lies to the type checker on purpose — the
    caller always reads the value through normal attribute or boolean access,
    which triggers resolution.
    """

    call = Call(
        kind="step",
        prompt=prompt,
        schema=schema,
        context=context,
        tools=tuple(tools),
        options={"label": label} if label is not None else {},
    )
    return cast("T", call)


def _dispatch_step(call: Call) -> Any:
    # Local import to avoid a cycle: the loop imports trace + runtime, both of
    # which the primitives module imports too.
    from ._internal.tool_loop import run_step

    client = current_model_client()
    if client is None:
        raise ConfigurationError(
            "No model client registered. Tests should call "
            "bridle.runtime.set_model_client(...) with a client; the Anthropic "
            "adapter wires this up automatically when used."
        )

    per_call_model = call.options.get("model")
    per_agent_model = call.options.get("agent_model") or current_agent_model()
    model = require_model(per_call=per_call_model, per_agent=per_agent_model)

    budget = effective_token_budget()
    if budget is not None and current_token_usage() >= budget:
        raise TokenBudgetExceededError(
            f"Token budget {budget} exhausted before step started.",
            used=current_token_usage(),
            budget=budget,
        )

    label = call.options.get("label") or call.prompt or "step"
    parent_trace = current_trace()
    trace = parent_trace if parent_trace is not None else Trace()
    trace_token = set_active_trace(trace) if parent_trace is None else None

    parent_id = current_event_id()
    start_event = Event.new("call_start", parent_id=parent_id, call_kind=call.kind, label=label)
    trace.emit(start_event)
    event_token = push_event_id(start_event.id)

    error: BaseException | None = None
    try:
        return run_step(
            prompt=call.prompt or "",
            schema=call.schema,
            context=call.context,
            tools=call.tools,
            client=client,
            model=model,
            max_turns=int(call.options.get("max_turns", 50)),
            max_schema_retries=int(call.options.get("max_schema_retries", 3)),
            parent_event_id=start_event.id,
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace.emit(
            Event.new(
                "call_end",
                call_kind=call.kind,
                parent_id=start_event.id,
                label=label,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )
        reset_event_id(event_token)
        if trace_token is not None:
            reset_active_trace(trace_token)


register("step", _dispatch_step)


def branch(
    prompt: str,
    *,
    schema: type[T] = bool,  # type: ignore[assignment]
    context: Any = None,
    label: str | None = None,
) -> T:
    """A step constrained to a single typed decision — no tools.

    The default *schema* is :class:`bool`; ``branch`` shines as the conditional
    in an ``if`` statement::

        if branch("is the evidence sufficient?", context=sources):
            ...

    For multi-way decisions, pass an enum or a ``Literal`` type as *schema*.
    """

    call = Call(
        kind="branch",
        prompt=prompt,
        schema=schema,
        context=context,
        tools=(),
        options={"label": label} if label is not None else {},
    )
    return cast("T", call)


# ``branch`` shares ``step``'s evaluator — same loop, different ``Call.kind``,
# which propagates through the trace as the ``call_kind`` on its events.
register("branch", _dispatch_step)


def loop(
    prompt: str,
    *,
    schema: type[T],
    until: Callable[[list[T]], bool],
    context: Any = None,
    tools: Sequence[Tool] = (),
    max_iterations: int = DEFAULT_LOOP_MAX_ITERATIONS,
    label: str | None = None,
) -> list[T]:
    """Repeatedly produce typed values until ``until(acc)`` is satisfied.

    Each iteration runs an inner ``step`` with the original *context* plus the
    running accumulator. ``until`` is a pure-Python predicate over the
    accumulator — it does not consult the model. When *max_iterations* is
    reached without satisfying *until*, ``loop`` raises
    :class:`bridle.errors.LoopExhaustedError`.
    """

    call = Call(
        kind="loop",
        prompt=prompt,
        schema=schema,
        context=context,
        tools=tuple(tools),
        options={
            "until": until,
            "max_iterations": int(max_iterations),
            **({"label": label} if label is not None else {}),
        },
    )
    return cast("list[T]", call)


def _dispatch_loop(call: Call) -> list[Any]:
    until: Callable[[list[Any]], bool] | None = call.options.get("until")
    if until is None or not callable(until):
        raise ConfigurationError("loop() requires an `until` predicate.")
    max_iterations = int(call.options.get("max_iterations", DEFAULT_LOOP_MAX_ITERATIONS))
    if max_iterations <= 0:
        raise ConfigurationError("loop() max_iterations must be positive.")

    label = call.options.get("label") or call.prompt or "loop"

    parent_trace = current_trace()
    trace = parent_trace if parent_trace is not None else Trace()
    trace_token = set_active_trace(trace) if parent_trace is None else None

    parent_id = current_event_id()
    start_event = Event.new("call_start", parent_id=parent_id, call_kind="loop", label=label)
    trace.emit(start_event)
    event_token = push_event_id(start_event.id)

    accumulator: list[Any] = []
    error: BaseException | None = None
    try:
        for iteration in range(max_iterations):
            iter_context = {
                "original_context": call.context,
                "previous_results": list(accumulator),
                "iteration": iteration,
            }
            sub_call = Call(
                kind="step",
                prompt=call.prompt or "",
                schema=call.schema,
                context=iter_context,
                tools=call.tools,
                options={"label": f"{label}[{iteration}]"},
            )
            item = resolve(sub_call)
            accumulator.append(item)
            if until(accumulator):
                return accumulator
        raise LoopExhaustedError(
            f"loop hit max_iterations={max_iterations} without satisfying the predicate.",
            iterations=max_iterations,
            accumulator=list(accumulator),
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        trace.emit(
            Event.new(
                "call_end",
                call_kind="loop",
                parent_id=start_event.id,
                label=label,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )
        reset_event_id(event_token)
        if trace_token is not None:
            reset_active_trace(trace_token)


register("loop", _dispatch_loop)


__all__ = ["branch", "loop", "step"]
