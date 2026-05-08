"""The wrapper algebra.

Each wrapper takes a :class:`Call` and returns a new :class:`Call` whose
``kind`` describes the wrapper. Resolution dispatches to the wrapper's
evaluator, which threads through to the inner call. Wrappers compose by
nesting — outer wraps inner — and read left-to-right exactly as written::

    cache(retry(timeout(step(...), seconds=10), attempts=3))

Order of evaluation: ``cache`` first, on miss ``retry``, each retry runs
``timeout`` around the inner ``step``.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from .call import Call, register, resolve
from .errors import BridleError
from .errors import TimeoutError as BridleTimeoutError
from .runtime import (
    current_cache,
    push_per_call_model,
    reset_per_call_model,
)
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit(trace: Trace | None, kind: Any, parent_id: str | None, payload: dict[str, Any]) -> None:
    if trace is None:
        return
    trace.emit(Event.new(kind, parent_id=parent_id, payload=payload))


def _wrap_evaluation(
    call: Call,
    *,
    label: str,
    inner_fn: Callable[[Trace, str], Any],
) -> Any:
    """Boilerplate around every wrapper dispatcher: trace, parent threading, errors.

    *inner_fn* receives the active trace and the wrapper's ``call_start`` id
    so it can emit wrapper-specific events with correct parentage.
    """

    parent_trace = current_trace()
    trace = parent_trace if parent_trace is not None else Trace()
    trace_token = set_active_trace(trace) if parent_trace is None else None

    parent_id = current_event_id()
    start_event = Event.new("call_start", parent_id=parent_id, call_kind=call.kind, label=label)
    trace.emit(start_event)
    event_token = push_event_id(start_event.id)

    error: BaseException | None = None
    try:
        return inner_fn(trace, start_event.id)
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


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def cache(
    call: Call,
    *,
    key: Callable[[Call], str] | str | None = None,
    backend: Any | None = None,
    ttl: float | None = None,
    label: str | None = None,
) -> Call:
    """Memoize *call*.

    On first evaluation the result is computed and stored. Subsequent
    evaluations against the same cache key return the stored value without
    re-running the inner work.

    *key* defaults to a deterministic hash of the inner call's kind, schema,
    context, prompt, and tools. Pass a string for a fixed key, or a callable
    for a custom derivation. *backend* defaults to the active backend
    registered via :func:`bridle.set_cache`, or an in-memory cache otherwise.

    Note: the default key includes the inner call's ``kind``, so
    ``cache(step(...))`` and ``cache(retry(step(...)))`` have distinct cache
    identities even when their underlying work is the same. To share a cache
    entry across wrapper compositions, pass an explicit ``key=`` string.
    """

    return Call(
        kind="cache",
        options={
            "inner": call,
            "key": key,
            "backend": backend,
            "ttl": ttl,
            "label": label or "cache",
        },
    )


def _dispatch_cache(call: Call) -> Any:
    from .cache import MISS, default_cache_key

    inner: Call = call.options["inner"]
    user_key = call.options.get("key")
    user_backend = call.options.get("backend")
    ttl = call.options.get("ttl")
    label = call.options.get("label", "cache")

    if isinstance(user_key, str):
        cache_key = user_key
    elif callable(user_key):
        cache_key = user_key(inner)
    else:
        cache_key = default_cache_key(inner)

    backend = user_backend or current_cache() or _get_default_memory_backend()

    def body(trace: Trace, start_id: str) -> Any:
        cached = backend.get(cache_key)
        if cached is not MISS:
            _emit(trace, "cache_hit", start_id, {"key": cache_key})
            return cached
        _emit(trace, "cache_miss", start_id, {"key": cache_key})
        result = resolve(inner.clone())
        try:
            backend.set(cache_key, result, ttl=ttl)
        except Exception as exc:
            _emit(trace, "cache_miss", start_id, {"key": cache_key, "write_error": str(exc)})
        return result

    return _wrap_evaluation(call, label=label, inner_fn=body)


_default_memory_backend: Any = None


def _get_default_memory_backend() -> Any:
    global _default_memory_backend
    if _default_memory_backend is None:
        from .cache.memory import MemoryCache

        _default_memory_backend = MemoryCache()
    return _default_memory_backend


register("cache", _dispatch_cache)


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def retry(
    call: Call,
    *,
    attempts: int = 3,
    on: type[BaseException] | tuple[type[BaseException], ...] = BridleError,
    backoff: float | Callable[[int], float] | None = None,
    label: str | None = None,
) -> Call:
    """Retry *call* up to *attempts* times when it raises *on*.

    Retries clone the inner call so each attempt resolves afresh. *backoff*
    can be a fixed number of seconds or a callable that takes the attempt
    index (0-based) and returns seconds. Exceptions outside *on* propagate
    immediately.
    """

    return Call(
        kind="retry",
        options={
            "inner": call,
            "attempts": int(attempts),
            "on": on,
            "backoff": backoff,
            "label": label or "retry",
        },
    )


def _dispatch_retry(call: Call) -> Any:
    import time

    inner: Call = call.options["inner"]
    attempts: int = int(call.options.get("attempts", 3))
    on = call.options.get("on", BridleError)
    backoff: float | Callable[[int], float] | None = call.options.get("backoff")
    label = call.options.get("label", "retry")

    if attempts <= 0:
        from .errors import ConfigurationError

        raise ConfigurationError("retry() attempts must be positive.")

    def body(trace: Trace, start_id: str) -> Any:
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                return resolve(inner.clone())
            except on as exc:  # type: ignore[misc]
                last_error = exc
                if attempt == attempts - 1:
                    break
                _emit(
                    trace,
                    "retry",
                    start_id,
                    {
                        "attempt": attempt + 1,
                        "remaining": attempts - attempt - 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if backoff is not None:
                    delay = backoff(attempt) if callable(backoff) else float(backoff)
                    if delay > 0:
                        time.sleep(delay)
        assert last_error is not None
        raise last_error

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("retry", _dispatch_retry)


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


def timeout(
    call: Call,
    *,
    seconds: float,
    label: str | None = None,
) -> Call:
    """Abort *call* if it takes longer than *seconds* wall-clock.

    Implementation uses a single-thread executor; on timeout the underlying
    work is abandoned (the thread keeps running) but the wrapper raises
    :class:`bridle.errors.TimeoutError`. For LLM calls, prefer the SDK's
    own ``timeout=`` parameter for hard cancellation; this wrapper is a
    coarser deadline at the program level.
    """

    return Call(
        kind="timeout",
        options={
            "inner": call,
            "seconds": float(seconds),
            "label": label or "timeout",
        },
    )


def _dispatch_timeout(call: Call) -> Any:
    import contextvars

    inner: Call = call.options["inner"]
    seconds: float = float(call.options["seconds"])
    label = call.options.get("label", "timeout")

    def body(_trace: Trace, _start_id: str) -> Any:
        # Copy the calling thread's context so the worker sees the same model
        # client, agent model, trace, etc. ``concurrent.futures`` does not
        # propagate :mod:`contextvars` automatically.
        ctx = contextvars.copy_context()

        def run_in_thread() -> Any:
            return ctx.run(lambda: resolve(inner.clone()))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(run_in_thread)
            try:
                return future.result(timeout=seconds)
            except concurrent.futures.TimeoutError as exc:
                raise BridleTimeoutError(f"call exceeded timeout of {seconds}s") from exc

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("timeout", _dispatch_timeout)


# ---------------------------------------------------------------------------
# with_model
# ---------------------------------------------------------------------------


def with_model(
    call: Call,
    model: str,
    *,
    label: str | None = None,
) -> Call:
    """Override the model used to evaluate *call*.

    Sets the per-call model layer (the highest-precedence layer of model
    resolution). Composes with any wrapper or primitive — the override
    propagates down to whichever inner step actually invokes the model.
    """

    return Call(
        kind="with_model",
        options={
            "inner": call,
            "model": model,
            "label": label or f"with_model[{model}]",
        },
    )


def _dispatch_with_model(call: Call) -> Any:
    inner: Call = call.options["inner"]
    model: str = call.options["model"]
    label = call.options.get("label", "with_model")

    def body(_trace: Trace, _start_id: str) -> Any:
        token = push_per_call_model(model)
        try:
            return resolve(inner.clone())
        finally:
            reset_per_call_model(token)

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("with_model", _dispatch_with_model)


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------


def fallback(
    call: Call,
    *alternates: Call,
    label: str | None = None,
) -> Call:
    """Try *call*; on failure, try each *alternates* in order.

    The first call that resolves without raising a :class:`bridle.BridleError`
    wins. If all options fail, the last error propagates. Useful for
    fallback model chains: ``fallback(with_model(step(...), "opus"),
    with_model(step(...), "sonnet"))``.
    """

    return Call(
        kind="fallback",
        options={
            "primary": call,
            "alternates": tuple(alternates),
            "label": label or "fallback",
        },
    )


def _dispatch_fallback(call: Call) -> Any:
    primary: Call = call.options["primary"]
    alternates: tuple[Call, ...] = call.options.get("alternates", ())
    label = call.options.get("label", "fallback")

    def body(trace: Trace, start_id: str) -> Any:
        candidates: tuple[Call, ...] = (primary, *alternates)
        last_error: BridleError | None = None
        for index, candidate in enumerate(candidates):
            try:
                return resolve(candidate.clone())
            except BridleError as exc:
                last_error = exc
                _emit(
                    trace,
                    "retry",
                    start_id,
                    {
                        "reason": "fallback",
                        "candidate": index,
                        "remaining": len(candidates) - index - 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
        assert last_error is not None
        raise last_error

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("fallback", _dispatch_fallback)


# ---------------------------------------------------------------------------
# mock
# ---------------------------------------------------------------------------


def mock(
    call: Call,
    value: Any,
    *,
    label: str | None = None,
) -> Call:
    """Replace *call*'s evaluation with a constant *value*.

    The wrapped call's structure is preserved in the trace, so test
    assertions on shape still pass; only the dispatch is short-circuited.
    Use in unit tests to substitute deterministic results for steps that
    would otherwise hit the model.
    """

    return Call(
        kind="mock",
        options={
            "inner": call,
            "value": value,
            "label": label or "mock",
        },
    )


def _dispatch_mock(call: Call) -> Any:
    label = call.options.get("label", "mock")

    def body(_trace: Trace, _start_id: str) -> Any:
        return call.options["value"]

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("mock", _dispatch_mock)


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def log(
    call: Call,
    *,
    level: LogLevel = "INFO",
    logger_name: str = "bridle",
    label: str | None = None,
) -> Call:
    """Stream the trace for *call*'s subtree to a Python logger.

    Subscribes a handler to the active trace before evaluation, unsubscribes
    after. Each event becomes a single log line with kind, call_kind, and
    label.
    """

    return Call(
        kind="log",
        options={
            "inner": call,
            "level": level,
            "logger_name": logger_name,
            "label": label or "log",
        },
    )


def _dispatch_log(call: Call) -> Any:
    inner: Call = call.options["inner"]
    level: LogLevel = cast("LogLevel", call.options.get("level", "INFO"))
    logger_name: str = cast("str", call.options.get("logger_name", "bridle"))
    label = call.options.get("label", "log")

    logger = logging.getLogger(logger_name)
    log_level = getattr(logging, level.upper(), logging.INFO)

    def body(trace: Trace, _start_id: str) -> Any:
        def handler(event: Event) -> None:
            logger.log(
                log_level,
                "bridle.%s call_kind=%s label=%s error=%s",
                event.kind,
                event.call_kind or "-",
                event.label or "-",
                event.error or "-",
            )

        unsubscribe = trace.subscribe(handler)
        try:
            return resolve(inner.clone())
        finally:
            unsubscribe()

    return _wrap_evaluation(call, label=label, inner_fn=body)


register("log", _dispatch_log)


__all__ = [
    "cache",
    "fallback",
    "log",
    "mock",
    "retry",
    "timeout",
    "with_model",
]
