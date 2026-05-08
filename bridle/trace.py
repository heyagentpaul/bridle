"""Execution trace: events, observers, and a context-bound active trace.

Every primitive emits structured events into the active :class:`Trace`. The
trace is the debugger — it captures every model turn, tool call, retry, cache
hit, and human response in order. Subscribers fire synchronously as events
land.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any, Literal

EventKind = Literal[
    "call_start",
    "call_end",
    "model_request",
    "model_response",
    "tool_call",
    "tool_result",
    "cache_hit",
    "cache_miss",
    "retry",
    "human_prompt",
    "human_response",
]


@dataclass(frozen=True)
class Event:
    """A single point on the trace.

    ``payload`` is event-specific (model params, tool args, cache key, etc.).
    ``error`` is set on ``call_end`` events when the call raised.
    """

    id: str
    parent_id: str | None
    kind: EventKind
    call_kind: str | None
    label: str | None
    timestamp: float
    duration_ms: float | None
    payload: dict[str, Any]
    error: str | None = None

    @staticmethod
    def new(
        kind: EventKind,
        *,
        parent_id: str | None = None,
        call_kind: str | None = None,
        label: str | None = None,
        duration_ms: float | None = None,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Event:
        return Event(
            id=uuid.uuid4().hex,
            parent_id=parent_id,
            kind=kind,
            call_kind=call_kind,
            label=label,
            timestamp=time.time(),
            duration_ms=duration_ms,
            payload=dict(payload or {}),
            error=error,
        )


Observer = Callable[[Event], None]


class Trace:
    """Ordered, observable list of :class:`Event`."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._observers: list[Observer] = []

    def emit(self, event: Event) -> None:
        self._events.append(event)
        # Snapshot observers so a subscriber added mid-emit doesn't fire on its
        # own registration event.
        for observer in tuple(self._observers):
            observer(event)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def subscribe(self, fn: Observer) -> Callable[[], None]:
        """Register *fn* as an observer. Returns a callable to unsubscribe."""

        self._observers.append(fn)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._observers.remove(fn)

        return _unsubscribe

    def to_dict(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._events]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(event)) for event in self._events)

    def tree(self) -> list[dict[str, Any]]:
        """Group events into a parent/children tree by ``parent_id``."""

        nodes: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for event in self._events:
            node: dict[str, Any] = {"event": asdict(event), "children": []}
            nodes[event.id] = node
            if event.parent_id and event.parent_id in nodes:
                children: list[dict[str, Any]] = nodes[event.parent_id]["children"]
                children.append(node)
            else:
                roots.append(node)
        return roots


_active_trace: ContextVar[Trace | None] = ContextVar("bridle_trace", default=None)
_current_event_id: ContextVar[str | None] = ContextVar("bridle_current_event_id", default=None)


def current_trace() -> Trace | None:
    """Return the trace active in this context, if any."""

    return _active_trace.get()


def set_active_trace(trace: Trace | None) -> Any:
    """Set the active trace for this context. Returns a token for ``reset``."""

    return _active_trace.set(trace)


def reset_active_trace(token: Any) -> None:
    _active_trace.reset(token)


def current_event_id() -> str | None:
    """Return the id of the enclosing call event, if any.

    A primitive's ``call_start`` event uses this as its ``parent_id`` to thread
    nesting through the trace. Composing agents pushes their ``call_start`` id
    so any inner ``step``/``branch``/``loop`` lands underneath it in
    :meth:`Trace.tree`.
    """

    return _current_event_id.get()


def push_event_id(event_id: str) -> Any:
    """Set the current parent event id. Returns a token for :func:`reset_event_id`."""

    return _current_event_id.set(event_id)


def reset_event_id(token: Any) -> None:
    _current_event_id.reset(token)


__all__ = [
    "Event",
    "EventKind",
    "Observer",
    "Trace",
    "current_event_id",
    "current_trace",
    "push_event_id",
    "reset_active_trace",
    "reset_event_id",
    "set_active_trace",
]
