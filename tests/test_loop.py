"""``loop`` primitive — repeat a step until a Python predicate is satisfied."""

from __future__ import annotations

from contextvars import copy_context
from typing import Any

import pytest
from pydantic import BaseModel

import bridle
from bridle import LoopExhaustedError, Trace, loop, resolve
from bridle.models import ModelResponse, ToolCall
from bridle.models.mock import MockModelClient, tool_response
from bridle.runtime import set_model_client
from bridle.trace import set_active_trace


class Source(BaseModel):
    url: str


def return_call(payload: dict[str, Any], call_id: str = "ret-1") -> ToolCall:
    return ToolCall(id=call_id, name="__bridle_return__", input=payload)


def _isolated(fn: Any) -> Any:
    return copy_context().run(fn)


def test_loop_collects_until_predicate_true() -> None:
    def body() -> None:
        # Three iterations, predicate stops after three.
        client = MockModelClient(
            [
                tool_response(return_call({"url": "u1"})),
                tool_response(return_call({"url": "u2"})),
                tool_response(return_call({"url": "u3"})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(
            loop(
                "find a source",
                schema=Source,
                until=lambda acc: len(acc) >= 3,
            )
        )
        assert [s.url for s in result] == ["u1", "u2", "u3"]

    _isolated(body)


def test_loop_stops_immediately_when_predicate_already_true_after_first() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"url": "only"}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(
            loop(
                "find a source",
                schema=Source,
                until=lambda acc: len(acc) >= 1,
            )
        )
        assert [s.url for s in result] == ["only"]

    _isolated(body)


def test_loop_raises_loop_exhausted_at_max_iterations() -> None:
    def body() -> None:
        # Predicate never satisfied — must hit max_iterations.
        client = MockModelClient(
            lambda _msgs, _tools: ModelResponse(
                text=None,
                tool_calls=(return_call({"url": "x"}),),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
            )
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(LoopExhaustedError) as exc_info:
            resolve(
                loop(
                    "find",
                    schema=Source,
                    until=lambda _: False,
                    max_iterations=4,
                )
            )

        err = exc_info.value
        assert err.iterations == 4
        assert err.accumulator is not None
        assert len(err.accumulator) == 4

    _isolated(body)


def test_loop_passes_accumulator_into_each_iteration_context() -> None:
    def body() -> None:
        seen_contexts: list[Any] = []

        def factory(messages: list[dict[str, Any]], _tools: Any) -> ModelResponse:
            # Capture the user message for inspection.
            seen_contexts.append(messages[0]["content"])
            n = len(seen_contexts)
            return ModelResponse(
                text=None,
                tool_calls=(return_call({"url": f"u{n}"}),),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
            )

        client = MockModelClient(factory)
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(
            loop(
                "find",
                schema=Source,
                until=lambda acc: len(acc) >= 3,
            )
        )

        # On iteration 2, the prompt should mention the accumulator from iter 1.
        assert "u1" in seen_contexts[1]
        # On iteration 3, both prior URLs are visible.
        assert "u1" in seen_contexts[2] and "u2" in seen_contexts[2]

    _isolated(body)


def test_loop_emits_call_start_and_call_end_with_loop_kind() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient([tool_response(return_call({"url": "u1"}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(
            loop(
                "find",
                schema=Source,
                until=lambda acc: len(acc) >= 1,
                label="search",
            )
        )

        kinds = [(e.kind, e.call_kind) for e in trace.events]
        assert ("call_start", "loop") in kinds
        assert ("call_end", "loop") in kinds

    _isolated(body)


def test_inner_step_events_have_loop_call_start_as_parent() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient(
            [
                tool_response(return_call({"url": "u1"})),
                tool_response(return_call({"url": "u2"})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(
            loop(
                "find",
                schema=Source,
                until=lambda acc: len(acc) >= 2,
            )
        )

        loop_start = next(
            e for e in trace.events if e.kind == "call_start" and e.call_kind == "loop"
        )
        step_starts = [e for e in trace.events if e.kind == "call_start" and e.call_kind == "step"]
        assert len(step_starts) == 2
        assert all(s.parent_id == loop_start.id for s in step_starts)

    _isolated(body)


def test_loop_propagates_step_failure() -> None:
    def body() -> None:
        # Force schema validation to fail every attempt → SchemaSatisfactionError.
        client = MockModelClient(
            [
                tool_response(return_call({"url": ["bad-list"]})),
                tool_response(return_call({"url": ["bad-list"]})),
                tool_response(return_call({"url": ["bad-list"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(bridle.SchemaSatisfactionError):
            resolve(
                loop(
                    "find",
                    schema=Source,
                    until=lambda acc: len(acc) >= 1,
                )
            )

    _isolated(body)


def test_zero_max_iterations_rejected() -> None:
    def body() -> None:
        client = MockModelClient([])
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(bridle.ConfigurationError):
            resolve(
                loop(
                    "find",
                    schema=Source,
                    until=lambda _: True,
                    max_iterations=0,
                )
            )

    _isolated(body)
