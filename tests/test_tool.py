"""``@tool`` decorator and :class:`Tool` semantics."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from bridle import Tool, tool


def test_bare_decorator_extracts_schema_from_type_hints() -> None:
    @tool
    def search(query: str, limit: int = 10) -> list[str]:
        """Web search.

        Returns up to ``limit`` URLs.
        """

        return [f"{query}-{i}" for i in range(limit)]

    assert isinstance(search, Tool)
    assert search.name == "search"
    assert search.description == "Web search."
    schema = search.parameters_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["properties"]["query"]["type"] == "string"
    assert "limit" in schema["properties"]
    assert schema["properties"]["limit"].get("default") == 10


def test_decorator_with_arguments_overrides_defaults() -> None:
    @tool(name="lookup", description="Custom description.", raise_on_error=True)
    def fn(id: str) -> str:
        return id

    assert fn.name == "lookup"
    assert fn.description == "Custom description."
    assert fn.raise_on_error is True


def test_no_args_function_yields_empty_schema() -> None:
    @tool
    def now() -> str:
        """Return the current ISO timestamp."""

        return "2026-05-07T00:00:00Z"

    assert now.parameters_schema["properties"] == {}


def test_tool_is_callable_for_unit_testing() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""

        return a + b

    assert add(2, 3) == 5
    # Same call via the wrapped fn:
    assert add.fn(4, 5) == 9


def test_complex_pydantic_argument_schema() -> None:
    class Filter(BaseModel):
        kind: str
        limit: int = 5

    @tool
    def query(f: Filter) -> list[str]:
        """Run a query."""

        return [f.kind] * f.limit

    schema = query.parameters_schema
    assert schema["type"] == "object"
    # Pydantic inlines the nested model under $defs or directly; either works.
    assert "f" in schema["properties"]


def test_missing_docstring_falls_back_to_function_name() -> None:
    @tool
    def silent(x: int) -> int:
        return x

    assert silent.description == "silent"


def test_explicit_description_wins_over_docstring() -> None:
    @tool(description="explicit")
    def documented(x: int) -> int:
        """ignored docstring."""

        return x

    assert documented.description == "explicit"


def test_decorator_handles_var_args() -> None:
    @tool
    def variadic(label: str, *args: Any, **kwargs: Any) -> str:
        return f"{label}:{len(args)}:{len(kwargs)}"

    # ``*args`` and ``**kwargs`` are excluded from the schema.
    schema = variadic.parameters_schema
    assert "label" in schema["properties"]
    assert "args" not in schema["properties"]
    assert "kwargs" not in schema["properties"]
    # The function still works when called directly.
    assert variadic("x", 1, 2, k="v") == "x:2:1"
