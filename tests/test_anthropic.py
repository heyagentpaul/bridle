"""Anthropic adapter — translation in both directions, end-to-end via a fake SDK."""

from __future__ import annotations

import os
from contextvars import copy_context
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import bridle
from bridle import resolve, step, tool
from bridle.models.anthropic import (
    AnthropicModelClient,
    _from_anthropic_response,
    _to_anthropic_tool,
    install,
)
from bridle.runtime import set_model_client

# -- Schema fixtures -----------------------------------------------------------


class Plan(BaseModel):
    topics: list[str]


# -- Fake SDK ------------------------------------------------------------------


def _block(kind: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=kind, **fields)


def _fake_response(
    *,
    content: list[SimpleNamespace],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


class _FakeMessages:
    def __init__(self, response_or_factory: Any) -> None:
        self._target = response_or_factory
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if callable(self._target):
            return self._target(**kwargs)
        return self._target


class _FakeAnthropic:
    def __init__(self, response_or_factory: Any) -> None:
        self.messages = _FakeMessages(response_or_factory)


def _isolated(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return copy_context().run(fn, *args, **kwargs)


# -- Translation tests ---------------------------------------------------------


def test_to_anthropic_tool_translates_fields() -> None:
    @tool
    def search(query: str, limit: int = 5) -> list[str]:
        """Web search."""

        return []

    out = _to_anthropic_tool(search)
    assert out["name"] == "search"
    assert out["description"] == "Web search."
    assert out["input_schema"]["type"] == "object"
    assert "query" in out["input_schema"]["properties"]


def test_from_anthropic_response_text_only() -> None:
    fake = _fake_response(
        content=[_block("text", text="hello there")],
        stop_reason="end_turn",
        input_tokens=4,
        output_tokens=8,
    )
    resp = _from_anthropic_response(fake)
    assert resp.text == "hello there"
    assert resp.tool_calls == ()
    assert resp.stop_reason == "end_turn"
    assert resp.input_tokens == 4
    assert resp.output_tokens == 8


def test_from_anthropic_response_tool_use_blocks() -> None:
    fake = _fake_response(
        content=[
            _block("text", text="thinking..."),
            _block("tool_use", id="t-1", name="search", input={"q": "weather"}),
            _block("tool_use", id="t-2", name="search", input={"q": "tides"}),
        ],
        stop_reason="tool_use",
    )
    resp = _from_anthropic_response(fake)
    assert resp.text == "thinking..."
    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].id == "t-1"
    assert resp.tool_calls[1].input == {"q": "tides"}
    assert resp.stop_reason == "tool_use"


def test_from_anthropic_response_unknown_stop_reason_maps_to_other() -> None:
    fake = _fake_response(content=[], stop_reason="weird_value")
    resp = _from_anthropic_response(fake)
    assert resp.stop_reason == "other"


# -- Client integration --------------------------------------------------------


def test_complete_passes_system_and_tools() -> None:
    @tool
    def search(query: str) -> list[str]:
        """Search."""

        return []

    fake = _FakeAnthropic(_fake_response(content=[_block("text", text="ok")]))
    client = AnthropicModelClient(client=fake, max_tokens=128)

    client.complete(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[search],
        system="be terse",
    )

    call_kwargs = fake.messages.calls[0]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 128
    assert call_kwargs["system"] == "be terse"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert isinstance(call_kwargs["tools"], list)
    assert call_kwargs["tools"][0]["name"] == "search"


def test_complete_without_tools_omits_tools_kwarg() -> None:
    fake = _FakeAnthropic(_fake_response(content=[_block("text", text="ok")]))
    client = AnthropicModelClient(client=fake)

    client.complete(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    call_kwargs = fake.messages.calls[0]
    # Anthropic rejects an empty tools list; the adapter should leave it out.
    assert "tools" not in call_kwargs


def test_complete_wraps_sdk_failures_in_model_error() -> None:
    class _Boom:
        class messages:  # noqa: N801 — mirroring the SDK shape
            @staticmethod
            def create(**_kwargs: Any) -> Any:
                raise RuntimeError("network down")

    client = AnthropicModelClient(client=_Boom())
    with pytest.raises(bridle.ModelError) as exc_info:
        client.complete(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "x"}],
            tools=[],
        )
    assert "network down" in str(exc_info.value)


# -- End-to-end through the tool loop ------------------------------------------


def test_step_runs_against_anthropic_adapter() -> None:
    """A real ``step`` driven by the adapter, with a fake SDK underneath."""

    def body() -> None:
        # Fake SDK that returns a __bridle_return__ tool call on the first turn.
        fake = _FakeAnthropic(
            _fake_response(
                content=[
                    _block(
                        "tool_use",
                        id="ret-1",
                        name="__bridle_return__",
                        input={"topics": ["a", "b"]},
                    ),
                ],
                stop_reason="tool_use",
                input_tokens=12,
                output_tokens=8,
            )
        )
        adapter = AnthropicModelClient(client=fake)
        set_model_client(adapter)
        bridle.configure(model="claude-sonnet-4-6")

        plan = resolve(step("draft", schema=Plan))
        assert plan.topics == ["a", "b"]

        # Verify the schema-tool reached the SDK as a real Anthropic tool block.
        sent_tools = fake.messages.calls[0]["tools"]
        names = [t["name"] for t in sent_tools]
        assert "__bridle_return__" in names

    _isolated(body)


def test_install_registers_adapter_globally() -> None:
    def body() -> None:
        fake = _FakeAnthropic(_fake_response(content=[_block("text", text="ok")]))
        adapter = install(client=fake)  # type: ignore[arg-type]
        from bridle.runtime import current_model_client

        assert current_model_client() is adapter

    _isolated(body)


# -- Live smoke (skipped without ANTHROPIC_API_KEY) ----------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No ANTHROPIC_API_KEY set — live test skipped.",
)
def test_live_smoke_against_real_api() -> None:  # pragma: no cover — environment-dependent
    from bridle.models.anthropic import install as install_real

    @bridle.agent(input=str, output=Plan, model="claude-haiku-4-5")
    def planner(topic: str) -> Plan:
        return step(
            "List exactly two short topic titles for a research brief on this subject.",
            schema=Plan,
            context=topic,
        )

    install_real()
    plan = bridle.resolve(planner("the weather on Mars"))
    assert isinstance(plan, Plan)
    assert len(plan.topics) >= 1
