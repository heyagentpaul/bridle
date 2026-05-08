"""``Call`` lazy resolution and dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bridle.call import Call, CallMeta, register, resolve, unregister


@pytest.fixture(autouse=True)
def _clean_dispatch():
    """Ensure each test starts with a clean dispatch table for the kinds it touches."""

    yield
    for kind in ("fake", "boolish", "iterable", "lengthy"):
        unregister(kind)


def test_call_construction_defaults() -> None:
    c = Call(kind="step", prompt="hi")
    assert c.kind == "step"
    assert c.prompt == "hi"
    assert c.tools == ()
    assert c.options == {}
    assert isinstance(c.meta, CallMeta)


def test_call_equality_is_identity_based() -> None:
    a = Call(kind="step", prompt="hi")
    b = Call(kind="step", prompt="hi")
    # Two distinct units of work; equality follows identity.
    assert a != b
    assert a == a


def test_evaluate_raises_without_dispatcher() -> None:
    c = Call(kind="never-registered")
    with pytest.raises(NotImplementedError) as exc_info:
        resolve(c)
    assert "never-registered" in str(exc_info.value)


def test_dispatcher_runs_via_resolve() -> None:
    register("fake", lambda call: f"resolved:{call.prompt}")
    c = Call(kind="fake", prompt="x")
    assert resolve(c) == "resolved:x"


def test_resolution_is_cached_on_first_use() -> None:
    calls: list[int] = []

    def dispatcher(_call: Call) -> int:
        calls.append(1)
        return 42

    register("fake", dispatcher)
    c = Call(kind="fake")
    assert resolve(c) == 42
    assert resolve(c) == 42  # second resolve must not redispatch
    assert len(calls) == 1


def test_getattr_triggers_evaluation() -> None:
    register("fake", lambda _call: SimpleNamespace(topics=["a", "b"]))
    c = Call(kind="fake")
    assert c.topics == ["a", "b"]


def test_dunder_attrs_do_not_trigger_evaluation() -> None:
    # Probes like repr / pickling hit ``__getattr__`` for dunder names; those
    # must not force evaluation.
    register("fake", lambda _call: pytest.fail("evaluation should not have fired"))
    c = Call(kind="fake")
    with pytest.raises(AttributeError):
        c.__some_dunder__  # noqa: B018


def test_bool_triggers_evaluation() -> None:
    register("boolish", lambda _call: True)
    register("falsy", lambda _call: 0)
    assert bool(Call(kind="boolish")) is True
    register("falsy", lambda _call: 0)
    register("boolish", lambda _call: False)
    assert bool(Call(kind="boolish")) is False
    unregister("falsy")


def test_iter_triggers_evaluation() -> None:
    register("iterable", lambda _call: [1, 2, 3])
    c = Call(kind="iterable")
    assert list(c) == [1, 2, 3]


def test_len_triggers_evaluation() -> None:
    register("lengthy", lambda _call: [10, 20, 30, 40])
    c = Call(kind="lengthy")
    assert len(c) == 4


def test_resolve_passes_through_non_calls() -> None:
    assert resolve(7) == 7
    assert resolve("hello") == "hello"
    assert resolve(None) is None


def test_with_options_returns_a_new_instance_with_merged_options() -> None:
    c = Call(kind="step", options={"a": 1})
    d = c.with_options(b=2)
    assert d is not c
    assert d.options == {"a": 1, "b": 2}
    # Original is untouched.
    assert c.options == {"a": 1}
