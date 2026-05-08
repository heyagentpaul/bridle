"""Translate a Pydantic ``ValidationError`` into a corrective user message.

When the model produces structured output that fails validation, we feed
the validation error back to the model as a tool result so it can correct
itself. The message has to be readable enough for the model to act on.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError


def correction_message(error: ValidationError, attempt: int, max_attempts: int) -> str:
    """Render *error* as a short, actionable user message."""

    issues: list[str] = []
    for entry in error.errors():
        loc_parts = cast("tuple[Any, ...]", entry.get("loc", ()) or ())
        loc = ".".join(str(part) for part in loc_parts) or "<root>"
        msg = str(entry.get("msg", "invalid value"))
        kind = str(entry.get("type", "error"))
        issues.append(f"  - {loc}: {msg} ({kind})")

    issue_text = "\n".join(issues) if issues else "  (no detail)"
    remaining = max_attempts - attempt
    plural = "attempt" if remaining == 1 else "attempts"

    return (
        f"Your response did not satisfy the schema. Issues:\n"
        f"{issue_text}\n"
        f"Try again. {remaining} {plural} remaining. "
        f"Call __bridle_return__ with a corrected payload."
    )


def short_error(error: Exception) -> str:
    """Render *error* compactly for embedding in tool-result messages."""

    return f"{type(error).__name__}: {error}"


def jsonable(value: Any) -> Any:
    """Best-effort conversion of *value* into something JSON-serialisable.

    Pydantic models go through ``model_dump``; primitives pass through;
    everything else falls back to ``repr``.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        items: list[Any] = list(cast("Any", value))
        return [jsonable(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        return {str(k): jsonable(v) for k, v in mapping.items()}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            return repr(value)
    return repr(value)


__all__ = ["correction_message", "jsonable", "short_error"]
