"""Anthropic adapter for the :class:`bridle.models.ModelClient` protocol.

The adapter is thin on purpose: translate Bridle's provider-neutral request
into Anthropic's ``messages.create`` arguments, dispatch to the SDK, and
read the response back into a :class:`ModelResponse`. The tool loop in
:mod:`bridle._internal.tool_loop` keeps the schema enforcement, retry
budgets, and trace emission centralised — the adapter does not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..errors import ModelError
from ..tool import Tool
from . import ModelResponse, StopReason, ToolCall

if TYPE_CHECKING:
    from anthropic import Anthropic


_DEFAULT_MAX_TOKENS = 4096


def _to_anthropic_tool(tool: Tool) -> dict[str, Any]:
    """Translate a Bridle :class:`Tool` into Anthropic's tool block schema."""

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters_schema,
    }


def _stop_reason(raw: Any) -> StopReason:
    """Map an Anthropic stop reason string into our typed literal."""

    if raw == "end_turn":
        return "end_turn"
    if raw == "tool_use":
        return "tool_use"
    if raw == "max_tokens":
        return "max_tokens"
    return "other"


def _from_anthropic_response(response: Any) -> ModelResponse:
    """Translate the SDK response into our :class:`ModelResponse`."""

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_chunks.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=str(getattr(block, "id", "")),
                    name=str(getattr(block, "name", "")),
                    input=dict(getattr(block, "input", {}) or {}),
                )
            )

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0)) if usage else 0
    output_tokens = int(getattr(usage, "output_tokens", 0)) if usage else 0

    return ModelResponse(
        text="".join(text_chunks) if text_chunks else None,
        tool_calls=tuple(tool_calls),
        stop_reason=_stop_reason(getattr(response, "stop_reason", None)),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class AnthropicModelClient:
    """Drives the Anthropic ``messages`` API on Bridle's behalf.

    Pass an existing :class:`anthropic.Anthropic` instance to share
    configuration (API key, base URL, custom HTTP client, retries) or let
    the adapter construct one with the SDK's default behavior.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        if client is None:
            try:
                from anthropic import Anthropic as _Anthropic
            except ImportError as exc:  # pragma: no cover — anthropic is a runtime dep
                raise ModelError(
                    "Bridle's Anthropic adapter requires the `anthropic` package."
                ) from exc
            client = _Anthropic()
        self._client: Any = client
        self._max_tokens = max_tokens

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[Tool],
        system: str | None = None,
        **params: Any,
    ) -> ModelResponse:
        max_tokens = int(params.pop("max_tokens", self._max_tokens))
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        if system is not None:
            kwargs["system"] = system
        anthropic_tools = [_to_anthropic_tool(t) for t in tools]
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        kwargs.update(params)

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"Anthropic request failed: {exc}") from exc

        return _from_anthropic_response(response)


def install(
    *,
    client: Anthropic | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> AnthropicModelClient:
    """Build and register an :class:`AnthropicModelClient` as the active client.

    Convenience for the common case::

        import bridle
        from bridle.models.anthropic import install
        install()
        bridle.configure(model="claude-sonnet-4-6")
    """

    from ..runtime import set_model_client

    adapter = AnthropicModelClient(client=cast("Any", client), max_tokens=max_tokens)
    set_model_client(adapter)
    return adapter


__all__ = ["AnthropicModelClient", "install"]
