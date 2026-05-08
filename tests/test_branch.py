"""``branch`` primitive — typed decisions, no tools."""

from __future__ import annotations

from contextvars import copy_context
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

import bridle
from bridle import Trace, branch, resolve
from bridle.models import ToolCall
from bridle.models.mock import MockModelClient, tool_response
from bridle.runtime import set_model_client
from bridle.trace import set_active_trace


def return_call(payload: dict[str, Any], call_id: str = "ret-1") -> ToolCall:
    return ToolCall(id=call_id, name="__bridle_return__", input=payload)


def _isolated(fn: Any) -> Any:
    return copy_context().run(fn)


def test_branch_default_bool_schema_returns_true() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"value": True}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = branch("is the answer yes?")
        assert resolve(result) is True

    _isolated(body)


def test_branch_default_bool_schema_returns_false_in_if_statement() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"value": False}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        if branch("is it raining?"):
            raise AssertionError("expected branch to be falsy")

    _isolated(body)


def test_branch_with_literal_schema() -> None:
    def body() -> None:
        # Literal schema: model picks one of three labels.
        choice_schema = Literal["yes", "no", "maybe"]

        client = MockModelClient([tool_response(return_call({"value": "maybe"}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = branch("decide", schema=choice_schema)  # type: ignore[arg-type]
        assert resolve(result) == "maybe"

    _isolated(body)


def test_branch_with_enum_schema() -> None:
    def body() -> None:
        class Verdict(StrEnum):
            ship = "ship"
            hold = "hold"

        client = MockModelClient([tool_response(return_call({"value": "ship"}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = branch("ship?", schema=Verdict)
        assert resolve(result) == Verdict.ship

    _isolated(body)


def test_branch_with_pydantic_schema() -> None:
    def body() -> None:
        class Decision(BaseModel):
            ship_it: bool
            confidence: float

        client = MockModelClient(
            [tool_response(return_call({"ship_it": True, "confidence": 0.85}))]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(branch("decide", schema=Decision))
        assert isinstance(result, Decision)
        assert result.ship_it is True
        assert result.confidence == 0.85

    _isolated(body)


def test_branch_emits_call_kind_branch_in_trace() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient([tool_response(return_call({"value": True}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(branch("yes?", label="root-branch"))

        kinds = [(e.kind, e.call_kind) for e in trace.events]
        assert ("call_start", "branch") in kinds
        assert ("call_end", "branch") in kinds

        start = next(e for e in trace.events if e.kind == "call_start" and e.call_kind == "branch")
        assert start.label == "root-branch"

    _isolated(body)


def test_branch_passes_no_tools_to_model() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"value": True}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(branch("yes?"))

        # Only the synthetic __bridle_return__ tool should be sent.
        sent_tools = client.calls[0]["tools"]
        names = [t.name for t in sent_tools]
        assert names == ["__bridle_return__"]

    _isolated(body)
