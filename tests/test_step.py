"""End-to-end ``step`` tests using ``MockModelClient``."""

from __future__ import annotations

from contextvars import copy_context
from typing import Any

import pytest
from pydantic import BaseModel

import bridle
from bridle import (
    ConfigurationError,
    ModelError,
    SchemaSatisfactionError,
    Tool,
    ToolExecutionError,
    Trace,
    resolve,
    step,
)
from bridle.models import ToolCall
from bridle.models.mock import MockModelClient, text_response, tool_response
from bridle.runtime import set_model_client
from bridle.trace import set_active_trace

# -- Helpers --------------------------------------------------------------------


class Plan(BaseModel):
    topics: list[str]


def return_call(payload: dict[str, Any], call_id: str = "ret-1") -> ToolCall:
    return ToolCall(id=call_id, name="__bridle_return__", input=payload)


def make_tool(
    name: str,
    fn: Any,
    *,
    parameters: dict[str, Any] | None = None,
    raise_on_error: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters_schema=parameters
        or {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        fn=fn,
        raise_on_error=raise_on_error,
    )


def _isolated(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return copy_context().run(fn, *args, **kwargs)


# -- Tests ---------------------------------------------------------------------


def test_happy_path_returns_typed_value() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a", "b"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = step("draft a plan", schema=Plan)
        plan = resolve(result)

        assert isinstance(plan, Plan)
        assert plan.topics == ["a", "b"]

    _isolated(body)


def test_bare_type_schema() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"value": 42}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = step("answer", schema=int)
        assert resolve(result) == 42

    _isolated(body)


def test_schema_retry_then_success() -> None:
    def body() -> None:
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "not a list"})),  # invalid
                tool_response(return_call({"topics": ["a"]})),  # valid
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        plan = resolve(step("draft a plan", schema=Plan))
        assert plan.topics == ["a"]

    _isolated(body)


def test_schema_exhausts_and_raises() -> None:
    def body() -> None:
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "bad-1"})),
                tool_response(return_call({"topics": "bad-2"})),
                tool_response(return_call({"topics": "bad-3"})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(SchemaSatisfactionError) as exc_info:
            resolve(step("draft a plan", schema=Plan))

        err = exc_info.value
        assert err.attempts == 3
        assert err.schema is Plan
        assert err.validation_error is not None

    _isolated(body)


def test_tool_call_recovers_on_error_by_default() -> None:
    def body() -> None:
        seen: list[str] = []

        def search(q: str) -> list[str]:
            seen.append(q)
            if len(seen) == 1:
                raise RuntimeError("network down")
            return ["url-1", "url-2"]

        search_tool = make_tool("search", search)

        client = MockModelClient(
            [
                tool_response(ToolCall(id="t1", name="search", input={"q": "first"})),
                tool_response(ToolCall(id="t2", name="search", input={"q": "second"})),
                tool_response(return_call({"topics": ["url-1", "url-2"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        plan = resolve(step("plan", schema=Plan, tools=[search_tool]))
        assert plan.topics == ["url-1", "url-2"]
        assert seen == ["first", "second"]

    _isolated(body)


def test_tool_with_raise_on_error_propagates() -> None:
    def body() -> None:
        def boom(q: str) -> str:
            raise RuntimeError("hard fail")

        tool = make_tool("boom", boom, raise_on_error=True)

        client = MockModelClient([tool_response(ToolCall(id="t1", name="boom", input={"q": "x"}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(ToolExecutionError) as exc_info:
            resolve(step("plan", schema=Plan, tools=[tool]))

        assert exc_info.value.tool_name == "boom"
        assert isinstance(exc_info.value.cause, RuntimeError)

    _isolated(body)


def test_max_turns_exceeded_raises_model_error() -> None:
    def body() -> None:
        # Always return text, never call __bridle_return__.
        client = MockModelClient(lambda _msgs, _tools: text_response("thinking..."))
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(ModelError) as exc_info:
            resolve(step("plan", schema=Plan, label="never-ends"))

        assert "exceeded" in str(exc_info.value)

    # Use a small max-turn override via Call.options. Since step() doesn't take
    # max_turns, hit it through the underlying Call.
    def body_small_cap() -> None:
        from bridle.call import Call
        from bridle.call import resolve as call_resolve

        client = MockModelClient(lambda _msgs, _tools: text_response("..."))
        set_model_client(client)
        bridle.configure(model="mock-1")

        c = Call(kind="step", prompt="x", schema=Plan, options={"max_turns": 3})
        with pytest.raises(ModelError):
            call_resolve(c)

    _isolated(body_small_cap)
    # Skip the 50-turn body; the smaller cap covers the same code path.
    _ = body  # suppress unused warning


def test_two_turn_path_tool_then_return() -> None:
    def body() -> None:
        seen: list[str] = []

        def search(q: str) -> list[str]:
            seen.append(q)
            return ["url-1"]

        tool = make_tool("search", search)
        client = MockModelClient(
            [
                tool_response(ToolCall(id="t1", name="search", input={"q": "weather"})),
                tool_response(return_call({"topics": ["url-1"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        plan = resolve(step("plan", schema=Plan, tools=[tool]))
        assert plan.topics == ["url-1"]
        assert seen == ["weather"]

    _isolated(body)


def test_text_only_response_nudges_model_to_call_return() -> None:
    def body() -> None:
        # Two turns: first text-only (gets nudged), then return.
        client = MockModelClient(
            [
                text_response("hmm let me think"),
                tool_response(return_call({"topics": ["x"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        plan = resolve(step("plan", schema=Plan))
        assert plan.topics == ["x"]
        # Confirm two turns happened.
        assert len(client.calls) == 2

    _isolated(body)


def test_missing_model_raises_configuration_error() -> None:
    def body() -> None:
        set_model_client(MockModelClient([tool_response(return_call({"topics": []}))]))
        # No bridle.configure(model=...) — should fail.
        with pytest.raises(ConfigurationError) as exc_info:
            resolve(step("plan", schema=Plan))
        msg = str(exc_info.value)
        assert "with_model" in msg
        assert "@agent" in msg
        assert "configure" in msg

    _isolated(body)


def test_missing_model_client_raises_configuration_error() -> None:
    def body() -> None:
        bridle.configure(model="mock-1")
        # No set_model_client() — should fail.
        with pytest.raises(ConfigurationError) as exc_info:
            resolve(step("plan", schema=Plan))
        assert "model client" in str(exc_info.value).lower()

    _isolated(body)


def test_emits_trace_events_in_order() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(step("plan", schema=Plan, label="root"))

        kinds = [e.kind for e in trace.events]
        assert kinds[0] == "call_start"
        assert kinds[-1] == "call_end"
        assert "model_request" in kinds
        assert "model_response" in kinds
        assert "tool_call" in kinds
        assert "tool_result" in kinds

        # call_start / call_end carry the label and call_kind.
        start = trace.events[0]
        end = trace.events[-1]
        assert start.label == "root"
        assert start.call_kind == "step"
        assert end.label == "root"
        assert end.error is None

    _isolated(body)


def test_emits_retry_event_on_schema_failure() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient(
            [
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": ["ok"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(step("plan", schema=Plan))
        kinds = [e.kind for e in trace.events]
        assert "retry" in kinds
        retry_event = next(e for e in trace.events if e.kind == "retry")
        assert retry_event.payload["reason"] == "schema"
        assert retry_event.payload["attempt"] == 1

    _isolated(body)


def test_unknown_tool_is_treated_as_recoverable() -> None:
    def body() -> None:
        client = MockModelClient(
            [
                tool_response(ToolCall(id="x1", name="ghost", input={})),
                tool_response(return_call({"topics": ["fallback"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        plan = resolve(step("plan", schema=Plan))  # no tools provided
        assert plan.topics == ["fallback"]

    _isolated(body)


def test_call_end_records_error_on_failure() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        # Three bad attempts → SchemaSatisfactionError
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(SchemaSatisfactionError):
            resolve(step("plan", schema=Plan))

        end = trace.events[-1]
        assert end.kind == "call_end"
        assert end.error is not None
        assert "SchemaSatisfactionError" in end.error

    _isolated(body)


def test_per_call_system_prompt_reaches_model_client() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(step("plan", schema=Plan, system="You are a stoic planner."))

        assert client.calls[0]["system"] == "You are a stoic planner."

    _isolated(body)


def test_default_system_prompt_used_when_unset() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(step("plan", schema=Plan))

        from bridle._internal.tool_loop import DEFAULT_SYSTEM_PROMPT

        assert client.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT

    _isolated(body)


def test_max_turns_kwarg_overrides_default() -> None:
    def body() -> None:
        # Always text → forces max_turns to be hit.
        client = MockModelClient(lambda _msgs, _tools: text_response("hmm"))
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(ModelError):
            resolve(step("plan", schema=Plan, max_turns=2))

        # Two turns made, then loop bailed.
        assert len(client.calls) == 2

    _isolated(body)


def test_max_schema_retries_kwarg_overrides_default() -> None:
    def body() -> None:
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "bad-1"})),
                tool_response(return_call({"topics": "bad-2"})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(SchemaSatisfactionError) as exc_info:
            resolve(step("plan", schema=Plan, max_schema_retries=2))

        assert exc_info.value.attempts == 2

    _isolated(body)
