"""Runtime configuration: contextvars, model resolution, error guidance."""

from __future__ import annotations

from contextvars import copy_context

import pytest

from bridle.errors import ConfigurationError
from bridle.runtime import (
    configure,
    current_cache,
    current_model,
    current_token_budget,
    require_model,
    set_cache,
)


def _isolated(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Run ``fn`` in a fresh context so other tests aren't polluted."""

    return copy_context().run(fn, *args, **kwargs)


def test_configure_sets_model_and_budget() -> None:
    def body() -> tuple[str | None, int | None]:
        configure(model="claude-sonnet-4-6", token_budget=10_000)
        return current_model(), current_token_budget()

    assert _isolated(body) == ("claude-sonnet-4-6", 10_000)


def test_configure_is_independent_arguments() -> None:
    def body() -> str | None:
        configure(model="claude-sonnet-4-6")
        # Calling configure again with only token_budget must not clear model.
        configure(token_budget=5_000)
        return current_model()

    assert _isolated(body) == "claude-sonnet-4-6"


def test_set_cache_round_trips() -> None:
    def body() -> object:
        set_cache("memory-backend")
        return current_cache()

    assert _isolated(body) == "memory-backend"


def test_require_model_raises_when_unset() -> None:
    def body() -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            require_model()
        message = str(exc_info.value)
        assert "with_model" in message
        assert "@agent" in message
        assert "configure" in message

    _isolated(body)


def test_require_model_resolution_order() -> None:
    def body() -> tuple[str, str, str]:
        configure(model="process-default")
        a = require_model()
        b = require_model(per_agent="agent-override")
        c = require_model(per_call="call-override", per_agent="agent-override")
        return a, b, c

    assert _isolated(body) == ("process-default", "agent-override", "call-override")


def test_contexts_do_not_bleed() -> None:
    # Configure in one context, observe absence in another.
    def setter() -> None:
        configure(model="ctx-a")

    def reader() -> str | None:
        return current_model()

    _isolated(setter)
    # Fresh context — should not see "ctx-a".
    assert _isolated(reader) is None
