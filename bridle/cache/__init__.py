"""Cache layer: backend protocol, key computation, missing sentinel.

Bridle ships caching as an opt-in wrapper, not a runtime feature. The
:func:`bridle.cache` wrapper consults the active backend before dispatching
its inner :class:`bridle.Call`, and writes the result back on a miss.

Backends implement a tiny synchronous interface — ``get``, ``set``,
``delete`` — keyed by string. Memory and file backends ship in v0.1.0; a
Redis backend is reserved as a stub and raises on construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from .._internal.json_repair import jsonable
from ..call import Call
from ..schema import fingerprint as schema_fingerprint
from ..tool import Tool


class _MissType:
    """Sentinel for ``not in cache``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<cache-miss>"


MISS: Any = _MissType()


@runtime_checkable
class CacheBackend(Protocol):
    """Synchronous get/set interface every backend implements."""

    def get(self, key: str) -> Any:
        """Return the cached value for *key*, or :data:`MISS` if absent or expired."""

        ...

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store *value* under *key*. ``ttl`` is in seconds; ``None`` means no expiry."""

        ...

    def delete(self, key: str) -> None:
        """Remove *key* from the backend. No-op if absent."""

        ...


def _stable_payload(value: Any) -> str:
    """Render *value* as a deterministic JSON string for hashing."""

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _tools_fingerprint(tools: tuple[Tool, ...]) -> str:
    if not tools:
        return ""
    parts = [f"{t.name}:{_stable_payload(t.parameters_schema)}" for t in tools]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def default_cache_key(call: Call) -> str:
    """Stable cache key for *call* — ``kind|schema|context|prompt|tools``.

    Two calls with the same kind, schema, context, prompt, and tools produce
    the same key — across processes, across runs.
    """

    schema_part = schema_fingerprint(call.schema) if call.schema is not None else ""
    parts = [
        f"kind={call.kind}",
        f"schema={schema_part}",
        f"prompt={call.prompt or ''}",
        f"context={_stable_payload(call.context)}",
        f"tools={_tools_fingerprint(call.tools)}",
    ]
    canonical = "\n".join(parts)
    return "bridle:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


__all__ = [
    "MISS",
    "CacheBackend",
    "default_cache_key",
]
