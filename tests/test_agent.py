"""``@agent`` decorator: composition, model propagation, validation, budget."""

from __future__ import annotations

from contextvars import copy_context
from typing import Any

import pytest
from pydantic import BaseModel

from bridle import (
    ConfigurationError,
    TokenBudgetExceededError,
    Trace,
    agent,
    resolve,
    step,
)
from bridle.models import ModelResponse, ToolCall
from bridle.models.mock import MockModelClient, tool_response
from bridle.runtime import set_model_client
from bridle.trace import set_active_trace

# -- Helpers --------------------------------------------------------------------


class Query(BaseModel):
    topic: str


class Plan(BaseModel):
    topics: list[str]


class Brief(BaseModel):
    headline: str


def return_call(payload: dict[str, Any], call_id: str = "ret-1") -> ToolCall:
    return ToolCall(id=call_id, name="__bridle_return__", input=payload)


def _isolated(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return copy_context().run(fn, *args, **kwargs)


# -- Tests ---------------------------------------------------------------------


def test_agent_runs_body_and_returns_value() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["x"]}))])
        set_model_client(client)

        result = resolve(planner(Query(topic="weather")))
        assert isinstance(result, Plan)
        assert result.topics == ["x"]

    _isolated(body)


def test_agent_propagates_model_to_inner_steps() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="agent-model")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["x"]}))])
        set_model_client(client)
        # Note: no bridle.configure(model=...) — the agent provides it.

        result = resolve(planner(Query(topic="x")))
        assert result.topics == ["x"]
        assert client.calls[0]["model"] == "agent-model"

    _isolated(body)


def test_per_call_model_overrides_agent_model() -> None:
    def body() -> None:
        from bridle.call import Call as RawCall
        from bridle.call import resolve as call_resolve

        @agent(input=Query, output=Plan, model="agent-model")
        def planner(q: Query) -> Plan:
            return call_resolve(
                RawCall(
                    kind="step",
                    prompt="plan",
                    schema=Plan,
                    context=q,
                    options={"model": "per-call-model"},
                )
            )

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(planner(Query(topic="x")))
        assert client.calls[0]["model"] == "per-call-model"

    _isolated(body)


def test_agent_validates_input_from_dict() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            assert isinstance(q, Query)
            assert q.topic == "rain"
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        # Pass a dict; the agent should coerce to Query.
        result = resolve(planner({"topic": "rain"}))
        assert result.topics == ["a"]

    _isolated(body)


def test_agent_validates_output() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            # Return a dict; agent should coerce to Plan.
            return {"topics": ["coerced"]}  # type: ignore[return-value]

        client = MockModelClient([])
        set_model_client(client)

        result = resolve(planner(Query(topic="x")))
        assert isinstance(result, Plan)
        assert result.topics == ["coerced"]

    _isolated(body)


def test_agent_emits_call_start_and_call_end() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q, label="inner")

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(planner(Query(topic="x")))

        kinds = [(e.kind, e.call_kind) for e in trace.events]
        assert ("call_start", "agent") in kinds
        assert ("call_end", "agent") in kinds
        assert ("call_start", "step") in kinds
        assert ("call_end", "step") in kinds

    _isolated(body)


def test_inner_step_event_parent_is_agent_call_start() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(planner(Query(topic="x")))

        agent_start = next(
            e for e in trace.events if e.kind == "call_start" and e.call_kind == "agent"
        )
        step_start = next(
            e for e in trace.events if e.kind == "call_start" and e.call_kind == "step"
        )
        assert step_start.parent_id == agent_start.id

    _isolated(body)


def test_nested_agents_keep_inner_trace_visible() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        @agent(input=Query, output=Plan, model="mock-1")
        def inner_planner(q: Query) -> Plan:
            return step("inner", schema=Plan, context=q, label="inner-step")

        @agent(input=Query, output=Plan, model="mock-1")
        def outer(q: Query) -> Plan:
            return inner_planner(q)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(outer(Query(topic="x")))

        agent_starts = [
            e for e in trace.events if e.kind == "call_start" and e.call_kind == "agent"
        ]
        assert len(agent_starts) == 2
        outer_evt, inner_evt = agent_starts
        assert outer_evt.parent_id is None
        assert inner_evt.parent_id == outer_evt.id

        step_start = next(
            e for e in trace.events if e.kind == "call_start" and e.call_kind == "step"
        )
        assert step_start.parent_id == inner_evt.id

    _isolated(body)


def test_agent_recursion_works() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            if q.topic == "second-pass":
                return step("plan", schema=Plan, context=q)
            return planner(Query(topic="second-pass"))

        client = MockModelClient([tool_response(return_call({"topics": ["recursed"]}))])
        set_model_client(client)

        result = resolve(planner(Query(topic="first")))
        assert result.topics == ["recursed"]

    _isolated(body)


def test_agent_without_schemas_runs_through() -> None:
    def body() -> None:
        @agent(model="mock-1")
        def passthrough(x: int) -> int:
            return x * 2

        result = resolve(passthrough(7))
        assert result == 14

    _isolated(body)


def test_agent_token_budget_raises_on_breach() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1", token_budget=10)
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        # The mock returns a response that uses 100 tokens — well over budget.
        client = MockModelClient(
            [
                ModelResponse(
                    text=None,
                    tool_calls=(return_call({"topics": ["x"]}),),
                    stop_reason="tool_use",
                    input_tokens=60,
                    output_tokens=40,
                )
            ]
        )
        set_model_client(client)

        with pytest.raises(TokenBudgetExceededError) as exc_info:
            resolve(planner(Query(topic="x")))
        assert exc_info.value.budget == 10
        assert exc_info.value.used >= 100

    _isolated(body)


def test_agent_input_dict_with_extra_keys_raises() -> None:
    from pydantic import ValidationError

    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        # Pass a non-coercible value — Query needs `topic`, we give wrong shape.
        with pytest.raises(ValidationError):
            resolve(planner({"unrelated": "field"}))

    _isolated(body)


def test_agent_propagates_failure() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            raise RuntimeError("body blew up")

        with pytest.raises(RuntimeError, match="body blew up"):
            resolve(planner(Query(topic="x")))

    _isolated(body)


def test_agent_call_end_records_error() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        @agent(input=Query, output=Plan, model="mock-1")
        def planner(q: Query) -> Plan:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            resolve(planner(Query(topic="x")))

        end = trace.events[-1]
        assert end.kind == "call_end"
        assert end.call_kind == "agent"
        assert end.error is not None
        assert "ValueError" in end.error

    _isolated(body)


def test_no_model_anywhere_raises_configuration_error() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan)  # no model on agent
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        # No bridle.configure(model=...) either.

        with pytest.raises(ConfigurationError):
            resolve(planner(Query(topic="x")))

    _isolated(body)


def test_agent_system_prompt_inherited_by_inner_steps() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1", system="You are agent voice.")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q)

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(planner(Query(topic="x")))
        assert client.calls[0]["system"] == "You are agent voice."

    _isolated(body)


def test_per_call_system_prompt_beats_per_agent() -> None:
    def body() -> None:
        @agent(input=Query, output=Plan, model="mock-1", system="agent voice")
        def planner(q: Query) -> Plan:
            return step("plan", schema=Plan, context=q, system="per-call voice")

        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)

        resolve(planner(Query(topic="x")))
        assert client.calls[0]["system"] == "per-call voice"

    _isolated(body)
