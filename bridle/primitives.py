"""The five primitives. Each returns a :class:`Call`.

This module wires up the dispatch table by registering each primitive's
evaluator with the :mod:`bridle.call` runtime at import time. Call
``bridle.step(...)`` to construct a step; the work happens when something
reads the result.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar, cast

from .call import Call, register
from .errors import ConfigurationError
from .runtime import current_model_client, require_model
from .tool import Tool
from .trace import Event, Trace, current_trace, reset_active_trace, set_active_trace

T = TypeVar("T")


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
    per_agent_model = call.options.get("agent_model")
    model = require_model(per_call=per_call_model, per_agent=per_agent_model)

    label = call.options.get("label") or call.prompt or "step"
    parent_trace = current_trace()
    trace = parent_trace if parent_trace is not None else Trace()
    token = set_active_trace(trace) if parent_trace is None else None

    start_event = Event.new("call_start", call_kind="step", label=label)
    trace.emit(start_event)
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
                call_kind="step",
                parent_id=start_event.id,
                label=label,
                error=f"{type(error).__name__}: {error}" if error is not None else None,
            )
        )
        if token is not None:
            reset_active_trace(token)


register("step", _dispatch_step)


__all__ = ["step"]
