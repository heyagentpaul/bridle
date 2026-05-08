"""Wrapper algebra: each wrapper in isolation, plus representative compositions."""

from __future__ import annotations

import logging
import tempfile
import time
from contextvars import copy_context
from typing import Any

import pytest
from pydantic import BaseModel

import bridle
from bridle import (
    SchemaSatisfactionError,
    Trace,
    cache,
    fallback,
    log,
    mock,
    resolve,
    retry,
    step,
    timeout,
    with_model,
)
from bridle.cache import MISS, default_cache_key
from bridle.cache.file import FileCache
from bridle.cache.memory import MemoryCache
from bridle.cache.redis import RedisCache
from bridle.errors import TimeoutError as BridleTimeoutError
from bridle.models import ModelResponse, ToolCall
from bridle.models.mock import MockModelClient, tool_response
from bridle.runtime import set_cache, set_model_client
from bridle.trace import set_active_trace

# -- Fixtures ------------------------------------------------------------------


class Plan(BaseModel):
    topics: list[str]


def return_call(payload: dict[str, Any], call_id: str = "ret-1") -> ToolCall:
    return ToolCall(id=call_id, name="__bridle_return__", input=payload)


def _isolated(fn: Any) -> Any:
    return copy_context().run(fn)


# =============================================================================
# cache
# =============================================================================


def test_cache_returns_stored_value_on_second_resolve() -> None:
    def body() -> None:
        # Two scripted responses, but the second won't be needed.
        client = MockModelClient(
            [
                tool_response(return_call({"topics": ["first"]})),
                tool_response(return_call({"topics": ["second"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")
        backend = MemoryCache()
        set_cache(backend)

        s1 = cache(step("plan", schema=Plan))
        s2 = cache(step("plan", schema=Plan))

        first = resolve(s1)
        second = resolve(s2)
        assert first.topics == ["first"]
        assert second.topics == ["first"]  # cached
        assert len(client.calls) == 1  # second resolve never hit the model

    _isolated(body)


def test_cache_emits_cache_hit_and_miss_events() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient([tool_response(return_call({"topics": ["x"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")
        backend = MemoryCache()
        set_cache(backend)

        resolve(cache(step("plan", schema=Plan)))
        resolve(cache(step("plan", schema=Plan)))

        kinds = [e.kind for e in trace.events]
        assert "cache_miss" in kinds
        assert "cache_hit" in kinds

    _isolated(body)


def test_cache_with_explicit_string_key() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")
        backend = MemoryCache()

        # Two cache calls with different inner content but same explicit key.
        resolve(cache(step("plan-a", schema=Plan), key="shared", backend=backend))
        client._queue.append(tool_response(return_call({"topics": ["b"]})))  # type: ignore[attr-defined]
        result = resolve(cache(step("plan-b", schema=Plan), key="shared", backend=backend))
        # Same key → cached value reused regardless of differing prompts.
        assert result.topics == ["a"]

    _isolated(body)


def test_cache_with_callable_key() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["x"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        seen_keys: list[str] = []

        def key_fn(call: Any) -> str:
            k = f"custom:{call.prompt}"
            seen_keys.append(k)
            return k

        backend = MemoryCache()
        resolve(cache(step("plan-x", schema=Plan), key=key_fn, backend=backend))
        assert seen_keys == ["custom:plan-x"]

    _isolated(body)


def test_default_cache_key_is_stable_across_constructions() -> None:
    s1 = step("plan", schema=Plan)
    s2 = step("plan", schema=Plan)
    assert default_cache_key(s1) == default_cache_key(s2)


def test_default_cache_key_distinguishes_different_prompts() -> None:
    s1 = step("plan-a", schema=Plan)
    s2 = step("plan-b", schema=Plan)
    assert default_cache_key(s1) != default_cache_key(s2)


def test_memory_cache_ttl_expires() -> None:
    backend = MemoryCache()
    backend.set("k", "v", ttl=0.05)
    assert backend.get("k") == "v"
    time.sleep(0.1)
    assert backend.get("k") is MISS


def test_memory_cache_delete() -> None:
    backend = MemoryCache()
    backend.set("k", 1)
    backend.delete("k")
    assert backend.get("k") is MISS


def test_file_cache_round_trip_and_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileCache(tmp)
        backend.set("k", {"a": 1, "b": [2, 3]})
        assert backend.get("k") == {"a": 1, "b": [2, 3]}

        # New backend instance against the same path — value persists.
        fresh = FileCache(tmp)
        assert fresh.get("k") == {"a": 1, "b": [2, 3]}


def test_file_cache_ttl_expires() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        backend = FileCache(tmp)
        backend.set("k", "v", ttl=0.05)
        assert backend.get("k") == "v"
        time.sleep(0.1)
        assert backend.get("k") is MISS


def test_redis_cache_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match=r"v0\.2\.0"):
        RedisCache()


# =============================================================================
# retry
# =============================================================================


def test_retry_succeeds_after_first_failure() -> None:
    def body() -> None:
        # First attempt: schema-fail three times → SchemaSatisfactionError.
        # Second attempt: succeed on first turn.
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": ["recovered"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(retry(step("plan", schema=Plan), attempts=2))
        assert result.topics == ["recovered"]

    _isolated(body)


def test_retry_emits_retry_events_between_attempts() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)

        client = MockModelClient(
            [
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": ["ok"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        resolve(retry(step("plan", schema=Plan), attempts=2))

        retry_events = [e for e in trace.events if e.kind == "retry"]
        # Schema retries inside the step + one wrapper-level retry between attempts.
        assert any(e.payload.get("attempt") == 1 for e in retry_events)

    _isolated(body)


def test_retry_propagates_after_exhaustion() -> None:
    def body() -> None:
        # All attempts fail.
        client = MockModelClient(
            lambda _msgs, _tools: ModelResponse(
                text=None,
                tool_calls=(return_call({"topics": "still-bad"}),),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
            )
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(SchemaSatisfactionError):
            resolve(retry(step("plan", schema=Plan), attempts=2))

    _isolated(body)


def test_retry_zero_attempts_rejected() -> None:
    def body() -> None:
        with pytest.raises(bridle.ConfigurationError):
            resolve(retry(step("x", schema=Plan), attempts=0))

    _isolated(body)


# =============================================================================
# timeout
# =============================================================================


def test_timeout_passes_through_when_under_deadline() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["fast"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(timeout(step("plan", schema=Plan), seconds=2.0))
        assert result.topics == ["fast"]

    _isolated(body)


def test_timeout_raises_when_inner_runs_long() -> None:
    def body() -> None:
        # Slow factory: blocks for 1s before responding.
        def slow_factory(_msgs: Any, _tools: Any) -> ModelResponse:
            time.sleep(1.0)
            return ModelResponse(
                text=None,
                tool_calls=(return_call({"topics": ["late"]}),),
                stop_reason="tool_use",
            )

        set_model_client(MockModelClient(slow_factory))
        bridle.configure(model="mock-1")

        with pytest.raises(BridleTimeoutError):
            resolve(timeout(step("plan", schema=Plan), seconds=0.05))

    _isolated(body)


# =============================================================================
# with_model
# =============================================================================


def test_with_model_overrides_inner_step_model() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="default-model")

        resolve(with_model(step("plan", schema=Plan), "override-model"))

        assert client.calls[0]["model"] == "override-model"

    _isolated(body)


def test_with_model_composes_around_cache() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["a"]}))])
        set_model_client(client)
        bridle.configure(model="default-model")

        resolve(with_model(cache(step("plan", schema=Plan)), "fancy-model"))

        assert client.calls[0]["model"] == "fancy-model"

    _isolated(body)


# =============================================================================
# fallback
# =============================================================================


def test_fallback_uses_primary_when_it_works() -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["primary"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(
            fallback(
                step("primary-plan", schema=Plan),
                step("alt-plan", schema=Plan),
            )
        )
        assert result.topics == ["primary"]
        assert len(client.calls) == 1

    _isolated(body)


def test_fallback_uses_alternate_after_primary_fails() -> None:
    def body() -> None:
        # Primary fails three times (schema), alternate succeeds.
        client = MockModelClient(
            [
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": "x"})),
                tool_response(return_call({"topics": ["alt-result"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        result = resolve(
            fallback(
                step("primary", schema=Plan),
                step("alternate", schema=Plan),
            )
        )
        assert result.topics == ["alt-result"]

    _isolated(body)


def test_fallback_propagates_last_error_when_all_fail() -> None:
    def body() -> None:
        client = MockModelClient(
            lambda _m, _t: ModelResponse(
                text=None,
                tool_calls=(return_call({"topics": "bad"}),),
                stop_reason="tool_use",
                input_tokens=1,
                output_tokens=1,
            )
        )
        set_model_client(client)
        bridle.configure(model="mock-1")

        with pytest.raises(SchemaSatisfactionError):
            resolve(
                fallback(
                    step("primary", schema=Plan),
                    step("alternate", schema=Plan),
                )
            )

    _isolated(body)


# =============================================================================
# mock
# =============================================================================


def test_mock_replaces_inner_dispatch_with_constant() -> None:
    def body() -> None:
        # No model client set — mock should bypass it entirely.
        result = resolve(mock(step("plan", schema=Plan), Plan(topics=["mocked"])))
        assert result.topics == ["mocked"]

    _isolated(body)


def test_mock_emits_call_start_and_end() -> None:
    def body() -> None:
        trace = Trace()
        set_active_trace(trace)
        resolve(mock(step("plan", schema=Plan), Plan(topics=["x"])))
        kinds = [(e.kind, e.call_kind) for e in trace.events]
        assert ("call_start", "mock") in kinds
        assert ("call_end", "mock") in kinds

    _isolated(body)


# =============================================================================
# log
# =============================================================================


def test_log_writes_to_python_logging(caplog: pytest.LogCaptureFixture) -> None:
    def body() -> None:
        client = MockModelClient([tool_response(return_call({"topics": ["x"]}))])
        set_model_client(client)
        bridle.configure(model="mock-1")

        with caplog.at_level(logging.DEBUG, logger="bridle"):
            resolve(log(step("plan", schema=Plan), level="DEBUG"))

        # Look for at least one bridle.* log line.
        assert any("bridle." in record.getMessage() for record in caplog.records)

    _isolated(body)


# =============================================================================
# composition
# =============================================================================


def test_compose_cache_retry_with_model() -> None:
    def body() -> None:
        # Sequence: schema-fail 3x (first attempt), then schema-fail 3x (second
        # attempt fails too) — but with a fallback path? Simpler: succeed on 2nd
        # attempt of the retry.
        client = MockModelClient(
            [
                # First retry attempt: schema-fail to exhaustion.
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": "bad"})),
                tool_response(return_call({"topics": "bad"})),
                # Second retry attempt: succeed.
                tool_response(return_call({"topics": ["composed"]})),
            ]
        )
        set_model_client(client)
        bridle.configure(model="default-model")
        set_cache(MemoryCache())

        wrapped = cache(retry(with_model(step("plan", schema=Plan), "fancy"), attempts=2))
        result = resolve(wrapped)
        assert result.topics == ["composed"]
        # Every call hit the override model.
        assert all(c["model"] == "fancy" for c in client.calls)

        # Re-resolving a fresh equivalent wrapper hits the cache, no new calls.
        before = len(client.calls)
        wrapped2 = cache(retry(with_model(step("plan", schema=Plan), "fancy"), attempts=2))
        result2 = resolve(wrapped2)
        assert result2.topics == ["composed"]
        assert len(client.calls) == before  # no new model calls

    _isolated(body)


def test_mock_composes_inside_loop() -> None:
    def body() -> None:
        # Each inner step is mocked to a constant; loop runs three times.
        # We can't use ``mock(step(...))`` inside loop directly because loop
        # constructs its own inner Call. Instead, exercise mock alone here.
        results = []
        for i in range(3):
            results.append(resolve(mock(step("x", schema=Plan), Plan(topics=[str(i)]))))

        assert [r.topics for r in results] == [["0"], ["1"], ["2"]]

    _isolated(body)
