"""Schema helpers: fingerprinting and bare-type wrapping.

The model layer always works with a Pydantic ``BaseModel``-derived schema.
Bare types (``bool``, ``int``, ``list[Source]``, etc.) get wrapped in a
synthetic root model so dispatch is uniform.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from pydantic import BaseModel, RootModel, TypeAdapter


def is_pydantic_model(t: Any) -> bool:
    """True when *t* is a Pydantic ``BaseModel`` subclass."""

    return isinstance(t, type) and issubclass(t, BaseModel)


def wrap_bare(t: Any) -> type[BaseModel]:
    """Wrap *t* in a synthetic ``RootModel`` if it isn't already a ``BaseModel``.

    Returns *t* unchanged when it's already a ``BaseModel`` subclass.
    """

    if is_pydantic_model(t):
        return cast("type[BaseModel]", t)

    # ``RootModel[t]`` produces a fresh class parameterised on the inner type.
    return RootModel[t]  # type: ignore[valid-type]


def schema_dump(t: Any) -> dict[str, Any]:
    """Return a JSON Schema dict for *t*, wrapping bare types as needed."""

    if is_pydantic_model(t):
        return cast("type[BaseModel]", t).model_json_schema()

    return TypeAdapter(t).json_schema()


def fingerprint(t: Any) -> str:
    """Deterministic SHA-256 of a schema's JSON Schema representation.

    Stable across processes for the same logical type. Used as part of the
    cache key and in trace events.
    """

    js = schema_dump(t)
    canonical = json.dumps(js, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "fingerprint",
    "is_pydantic_model",
    "schema_dump",
    "wrap_bare",
]
