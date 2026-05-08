"""Process and context configuration.

Bridle ships zero defaults. The user supplies a model at one of three layers
— per-call (``with_model``), per-agent (``@agent(model=...)``), or process-wide
(``configure(model=...)``). When none of the three is set, the runtime raises
:class:`bridle.errors.ConfigurationError` with explicit guidance.

Configuration is held in :mod:`contextvars` so concurrent agents in the same
process can carry their own settings without colliding.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from .errors import ConfigurationError

if TYPE_CHECKING:
    pass


HumanChannel = Callable[[str], str]


_model: ContextVar[str | None] = ContextVar("bridle_model", default=None)
_token_budget: ContextVar[int | None] = ContextVar("bridle_token_budget", default=None)
_cache_backend: ContextVar[Any | None] = ContextVar("bridle_cache_backend", default=None)
_human_channel: ContextVar[HumanChannel | None] = ContextVar("bridle_human_channel", default=None)
_model_client: ContextVar[Any | None] = ContextVar("bridle_model_client", default=None)
_agent_model: ContextVar[str | None] = ContextVar("bridle_agent_model", default=None)
_agent_token_budget: ContextVar[int | None] = ContextVar("bridle_agent_token_budget", default=None)
_token_usage: ContextVar[int] = ContextVar("bridle_token_usage", default=0)


def configure(
    *,
    model: str | None = None,
    token_budget: int | None = None,
    cache: Any | None = None,
    model_client: Any | None = None,
) -> None:
    """Set process-wide defaults. Each argument is independent and opt-in."""

    if model is not None:
        _model.set(model)
    if token_budget is not None:
        _token_budget.set(token_budget)
    if cache is not None:
        _cache_backend.set(cache)
    if model_client is not None:
        _model_client.set(model_client)


def set_human_channel(fn: HumanChannel) -> None:
    """Register a function that prompts a human and returns their response."""

    _human_channel.set(fn)


def set_cache(backend: Any) -> None:
    """Register the active cache backend."""

    _cache_backend.set(backend)


def set_model_client(client: Any) -> None:
    """Register the active model client. Used by tests and the Anthropic adapter."""

    _model_client.set(client)


def current_model() -> str | None:
    return _model.get()


def current_token_budget() -> int | None:
    return _token_budget.get()


def current_cache() -> Any | None:
    return _cache_backend.get()


def current_human_channel() -> HumanChannel | None:
    return _human_channel.get()


def current_model_client() -> Any | None:
    return _model_client.get()


def current_agent_model() -> str | None:
    """Model declared by the enclosing ``@agent``, if any."""

    return _agent_model.get()


def push_agent_model(model: str | None) -> Any:
    return _agent_model.set(model)


def reset_agent_model(token: Any) -> None:
    _agent_model.reset(token)


def current_agent_token_budget() -> int | None:
    """Token budget declared by the enclosing ``@agent``, if any."""

    return _agent_token_budget.get()


def push_agent_token_budget(budget: int | None) -> Any:
    return _agent_token_budget.set(budget)


def reset_agent_token_budget(token: Any) -> None:
    _agent_token_budget.reset(token)


def current_token_usage() -> int:
    """Cumulative tokens used in this context."""

    return _token_usage.get()


def push_token_usage(value: int) -> Any:
    return _token_usage.set(value)


def reset_token_usage(token: Any) -> None:
    _token_usage.reset(token)


def bump_token_usage(delta: int) -> int:
    """Add *delta* to the current token usage. Returns the new total."""

    new_total = _token_usage.get() + max(0, delta)
    _token_usage.set(new_total)
    return new_total


def effective_token_budget() -> int | None:
    """The active budget — agent-level wins over process-level."""

    return current_agent_token_budget() or current_token_budget()


def require_model(per_call: str | None = None, per_agent: str | None = None) -> str:
    """Resolve the active model name, in order: per-call, per-agent, process.

    Raises :class:`ConfigurationError` with explicit guidance when nothing is set.
    """

    model = per_call or per_agent or current_agent_model() or current_model()
    if model is None:
        raise ConfigurationError(
            "No model specified. Set one of:\n"
            "  - per-call:    with_model(call, '<model-id>')\n"
            "  - per-agent:   @agent(..., model='<model-id>')\n"
            "  - process:     bridle.configure(model='<model-id>')"
        )
    return model


__all__ = [
    "HumanChannel",
    "bump_token_usage",
    "configure",
    "current_agent_model",
    "current_agent_token_budget",
    "current_cache",
    "current_human_channel",
    "current_model",
    "current_model_client",
    "current_token_budget",
    "current_token_usage",
    "effective_token_budget",
    "push_agent_model",
    "push_agent_token_budget",
    "push_token_usage",
    "require_model",
    "reset_agent_model",
    "reset_agent_token_budget",
    "reset_token_usage",
    "set_cache",
    "set_human_channel",
    "set_model_client",
]
