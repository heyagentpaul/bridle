"""The lazy ``Call`` value and its evaluator.

Primitives (``step``, ``branch``, ``loop``) and
the wrapper algebra (``cache``, ``retry``, ``timeout``, ...) all return a
:class:`Call`. A ``Call`` describes work to be done, not the result of doing
it. It evaluates lazily on first use — when its truthiness, iterability, or
an attribute is read — or explicitly via :func:`resolve`.

Evaluation is driven by a small dispatch table keyed on ``Call.kind``. Each
primitive registers its dispatcher when its module loads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallMeta:
    """Provenance metadata for a :class:`Call`."""

    label: str | None = None
    parent_id: str | None = None
    origin: str | None = None


class _Unset:
    """Sentinel for "not yet resolved" — distinct from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


@dataclass(eq=False)
class Call:
    """A lazy, typed description of agent work.

    Treat instances as immutable. Wrappers return new ``Call``\\ s rather than
    mutating in place. Equality is identity-based — two ``Call``\\ s with the
    same fields are not "equal" because semantically each represents a
    distinct unit of work.

    Watch out: a ``Call`` is truthy on resolution, not on existence. Writing
    ``if some_step:`` will *evaluate* ``some_step`` to decide the branch —
    which means hitting the model. This is intentional (it's how
    :func:`bridle.branch` reads naturally as ``if branch(...)``) but it
    surprises people checking "did I get a call back yet?" Use ``is None`` /
    ``is not None`` for existence checks; reserve truthiness for when you
    actually want resolution. The same applies to :meth:`__iter__`,
    :meth:`__len__`, and attribute access.
    """

    kind: str
    prompt: str | None = None
    schema: Any = None
    context: Any = None
    tools: tuple[Any, ...] = ()
    options: dict[str, Any] = field(default_factory=dict[str, Any])
    meta: CallMeta = field(default_factory=CallMeta)

    def __post_init__(self) -> None:
        # Cached evaluation result; not a dataclass field so it stays out of
        # ``__init__`` and equality.
        self._resolved: Any = _UNSET

    # --- Lazy resolution surface --------------------------------------------------

    def __bool__(self) -> bool:
        """Resolve the call and return the truthiness of its result.

        Watch out: this triggers full evaluation. ``if some_step:`` runs the
        step. That's intentional — it's what makes ``if branch("..."):`` read
        naturally — but it's a footgun for anyone testing whether a ``Call``
        exists. Use ``is None`` / ``is not None`` for existence checks.
        """

        return bool(self._resolve())

    def __iter__(self) -> Iterator[Any]:
        return iter(self._resolve())

    def __len__(self) -> int:
        value = self._resolve()
        return len(value)

    def __getattr__(self, name: str) -> Any:
        # ``__getattr__`` only fires for names not found by normal lookup, so
        # dataclass fields and ``_resolved`` resolve before reaching here.
        # Refuse dunders so Python's introspection (repr, pickling, copy) does
        # not trigger evaluation by accident.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    def _resolve(self) -> Any:
        if self._resolved is _UNSET:
            self._resolved = evaluate(self)
        return self._resolved

    # --- Public helpers ------------------------------------------------------------

    def with_options(self, **updates: Any) -> Call:
        """Return a new ``Call`` with merged ``options``."""

        return Call(
            kind=self.kind,
            prompt=self.prompt,
            schema=self.schema,
            context=self.context,
            tools=self.tools,
            options={**self.options, **updates},
            meta=self.meta,
        )

    def clone(self) -> Call:
        """Return an unevaluated copy of this :class:`Call`.

        Wrappers that re-evaluate the same logical call multiple times
        (``retry``, ``fallback``, etc.) clone first so each attempt
        resolves afresh — the original's ``_resolved`` cache is left
        untouched.
        """

        return Call(
            kind=self.kind,
            prompt=self.prompt,
            schema=self.schema,
            context=self.context,
            tools=self.tools,
            options={**self.options},
            meta=self.meta,
        )


# --- Dispatch -------------------------------------------------------------------

Dispatcher = Callable[[Call], Any]
_dispatch: dict[str, Dispatcher] = {}


def register(kind: str, dispatcher: Dispatcher) -> None:
    """Register *dispatcher* as the evaluator for ``Call.kind == kind``.

    Re-registering replaces the existing dispatcher. Used by primitives and
    by tests that need to substitute behavior.
    """

    _dispatch[kind] = dispatcher


def unregister(kind: str) -> None:
    _dispatch.pop(kind, None)


def evaluate(call: Call) -> Any:
    """Resolve *call* by routing to its registered dispatcher."""

    dispatcher = _dispatch.get(call.kind)
    if dispatcher is None:
        raise NotImplementedError(
            f"No dispatcher registered for Call.kind={call.kind!r}. "
            "Primitives register their dispatchers at import time."
        )
    return dispatcher(call)


def resolve(call: Any) -> Any:
    """Force evaluation of *call* if it's a :class:`Call`; otherwise return as-is."""

    if isinstance(call, Call):
        return call._resolve()  # pyright: ignore[reportPrivateUsage]
    return call


__all__ = [
    "Call",
    "CallMeta",
    "Dispatcher",
    "evaluate",
    "register",
    "resolve",
    "unregister",
]
