"""The synthetic ``__bridle_return__`` tool that carries the step's typed return.

The model satisfies a step by calling this tool with arguments that match the
step's schema. Bare types (``bool``, ``int``, ``list[Source]``) get wrapped
in a synthetic root model with a single ``value`` field so the tool always
exposes an object-shaped JSON Schema, which is what providers require for
tool parameters.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model

from ..schema import is_pydantic_model
from ..tool import Tool

RETURN_TOOL_NAME = "__bridle_return__"
RETURN_TOOL_DESCRIPTION = (
    "Submit the final answer for this step. Call exactly once with arguments "
    "matching the requested schema. Do not call any other tool after this one."
)
_BARE_VALUE_FIELD = "value"


def _input_model(schema: Any) -> type[BaseModel]:
    """The Pydantic model the tool's ``input`` is validated against.

    For Pydantic ``BaseModel`` schemas, that's the schema itself. For bare
    types, we synthesise ``{ "value": T }``.
    """

    if is_pydantic_model(schema):
        return cast("type[BaseModel]", schema)
    return create_model("BridleReturn", **{_BARE_VALUE_FIELD: (schema, ...)})  # type: ignore[call-overload]


def make_return_tool(schema: Any) -> Tool:
    """Build the synthetic ``__bridle_return__`` tool for *schema*."""

    model = _input_model(schema)
    params = model.model_json_schema()
    # Anthropic and OpenAI both expect an object schema; Pydantic produces one.
    if params.get("type") != "object":  # pragma: no cover — defensive
        params = {"type": "object", "properties": params.get("properties", {})}

    def _never_called(**_kwargs: Any) -> Any:  # pragma: no cover — model never executes this
        raise RuntimeError("The return tool is intercepted by the tool loop and never executed.")

    return Tool(
        name=RETURN_TOOL_NAME,
        description=RETURN_TOOL_DESCRIPTION,
        parameters_schema=params,
        fn=_never_called,
        raise_on_error=False,
        metadata={"synthetic": True, "schema": schema},
    )


def validate_return(schema: Any, payload: dict[str, Any]) -> Any:
    """Validate the model's ``__bridle_return__`` arguments against *schema*.

    Returns the typed value: a Pydantic instance for model schemas, the bare
    ``value`` for wrapped types. Raises :class:`pydantic.ValidationError` on
    failure — the tool loop converts that into a corrective retry.
    """

    model = _input_model(schema)
    instance = model.model_validate(payload)
    if is_pydantic_model(schema):
        return instance
    return getattr(instance, _BARE_VALUE_FIELD)


def is_return_call(name: str) -> bool:
    return name == RETURN_TOOL_NAME


__all__ = [
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_NAME",
    "ValidationError",
    "is_return_call",
    "make_return_tool",
    "validate_return",
]
