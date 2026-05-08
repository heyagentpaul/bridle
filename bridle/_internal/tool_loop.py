"""The inner tool-call loop for a single :func:`step`.

Drives a model client through a sequence of turns: send messages, read the
response, execute any tools the model called, feed results back, and stop
when the model calls ``__bridle_return__`` with arguments that satisfy the
step's schema.

Two retry budgets share the loop:

* ``max_turns`` caps the total number of model turns. A runaway loop hits
  this and raises :class:`bridle.errors.ModelError`.
* ``max_schema_retries`` caps the number of times the model can produce
  invalid structured output before :class:`bridle.errors.SchemaSatisfactionError`
  fires. Validation errors are fed back to the model as corrective tool
  results so it can adjust.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from ..errors import (
    ModelError,
    SchemaSatisfactionError,
    TokenBudgetExceededError,
    ToolExecutionError,
)
from ..models import ModelClient, ModelResponse
from ..runtime import bump_token_usage, current_token_usage, effective_token_budget
from ..tool import Tool
from ..trace import Event, current_trace
from .json_repair import correction_message, jsonable, short_error
from .return_tool import is_return_call, make_return_tool, validate_return

DEFAULT_MAX_TURNS = 50
DEFAULT_MAX_SCHEMA_RETRIES = 3
DEFAULT_SYSTEM_PROMPT = (
    "You are completing one step of a structured program. Use the tools "
    "provided when they help. When you have the final answer, call "
    "__bridle_return__ with arguments that match its schema."
)


class _Missing:
    __slots__ = ()


_MISSING: Any = _Missing()


def run_step(
    *,
    prompt: str,
    schema: Any,
    context: Any,
    tools: Sequence[Tool],
    client: ModelClient,
    model: str,
    system: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_schema_retries: int = DEFAULT_MAX_SCHEMA_RETRIES,
    parent_event_id: str | None = None,
) -> Any:
    """Run one ``step`` to completion.

    Returns the typed value the model produced via ``__bridle_return__``.
    Raises one of :class:`SchemaSatisfactionError`, :class:`ToolExecutionError`,
    or :class:`ModelError`.
    """

    return_tool = make_return_tool(schema)
    all_tools: tuple[Tool, ...] = (*tools, return_tool)
    user_tools_by_name = {tool.name: tool for tool in tools}

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _user_prompt(prompt, context)},
    ]
    schema_attempts = 0
    trace = current_trace()

    for turn in range(max_turns):
        _emit(
            trace,
            "model_request",
            parent_event_id,
            {"turn": turn, "model": model, "tools": [t.name for t in all_tools]},
        )
        response = client.complete(
            model=model,
            messages=messages,
            tools=all_tools,
            system=system or DEFAULT_SYSTEM_PROMPT,
        )
        _emit(
            trace,
            "model_response",
            parent_event_id,
            {
                "turn": turn,
                "stop_reason": response.stop_reason,
                "tool_calls": [tc.name for tc in response.tool_calls],
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

        bump_token_usage(response.input_tokens + response.output_tokens)
        budget = effective_token_budget()
        if budget is not None and current_token_usage() > budget:
            raise TokenBudgetExceededError(
                f"Token budget {budget} exhausted mid-step.",
                used=current_token_usage(),
                budget=budget,
            )

        messages.append({"role": "assistant", "content": _assistant_content(response)})

        if not response.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the __bridle_return__ tool to submit the final answer for this step."
                    ),
                }
            )
            continue

        user_blocks: list[dict[str, Any]] = []
        return_value: Any = _MISSING
        last_validation_error: ValidationError | None = None
        last_return_call_id: str | None = None

        for tc in response.tool_calls:
            if is_return_call(tc.name):
                _emit(
                    trace,
                    "tool_call",
                    parent_event_id,
                    {"name": tc.name, "input": jsonable(tc.input)},
                )
                try:
                    value = validate_return(schema, tc.input)
                except ValidationError as exc:
                    last_validation_error = exc
                    last_return_call_id = tc.id
                    _emit(
                        trace,
                        "tool_result",
                        parent_event_id,
                        {"name": tc.name, "error": short_error(exc)},
                    )
                    continue
                _emit(
                    trace,
                    "tool_result",
                    parent_event_id,
                    {"name": tc.name, "ok": True},
                )
                if return_value is _MISSING:
                    return_value = value
                continue

            tool = user_tools_by_name.get(tc.name)
            _emit(
                trace,
                "tool_call",
                parent_event_id,
                {"name": tc.name, "input": jsonable(tc.input)},
            )
            if tool is None:
                err = f"Unknown tool: {tc.name!r}"
                user_blocks.append(_tool_result_block(tc.id, err, is_error=True))
                _emit(trace, "tool_result", parent_event_id, {"name": tc.name, "error": err})
                continue

            try:
                result = tool.fn(**tc.input)
            except Exception as exc:
                if tool.raise_on_error:
                    raise ToolExecutionError(
                        f"Tool {tc.name!r} raised", tool_name=tc.name, cause=exc
                    ) from exc
                err_text = short_error(exc)
                user_blocks.append(_tool_result_block(tc.id, err_text, is_error=True))
                _emit(
                    trace,
                    "tool_result",
                    parent_event_id,
                    {"name": tc.name, "error": err_text},
                )
                continue

            user_blocks.append(_tool_result_block(tc.id, jsonable(result), is_error=False))
            _emit(
                trace,
                "tool_result",
                parent_event_id,
                {"name": tc.name, "ok": True},
            )

        if return_value is not _MISSING:
            return return_value

        if last_validation_error is not None:
            schema_attempts += 1
            if schema_attempts >= max_schema_retries:
                raise SchemaSatisfactionError(
                    "Model failed to produce schema-valid output after "
                    f"{schema_attempts} attempts.",
                    schema=schema,
                    last_attempt=jsonable(_first_return_input(response)),
                    validation_error=last_validation_error,
                    attempts=schema_attempts,
                )
            _emit(
                trace,
                "retry",
                parent_event_id,
                {"reason": "schema", "attempt": schema_attempts},
            )
            user_blocks.append(
                _tool_result_block(
                    last_return_call_id or "__bridle_return__",
                    correction_message(last_validation_error, schema_attempts, max_schema_retries),
                    is_error=True,
                )
            )

        if user_blocks:
            messages.append({"role": "user", "content": user_blocks})

    raise ModelError(f"step exceeded max turns ({max_turns}) without a final answer")


def _emit(
    trace: Any,
    kind: str,
    parent_id: str | None,
    payload: dict[str, Any],
) -> None:
    if trace is None:
        return
    trace.emit(
        Event.new(
            kind,  # type: ignore[arg-type]
            parent_id=parent_id,
            payload=payload,
        )
    )


def _assistant_content(response: ModelResponse) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if response.text:
        blocks.append({"type": "text", "text": response.text})
    for tc in response.tool_calls:
        blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": dict(tc.input)})
    return blocks


def _tool_result_block(tool_use_id: str, content: Any, *, is_error: bool) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(jsonable(content), default=str)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }


def _first_return_input(response: ModelResponse) -> Any:
    for tc in response.tool_calls:
        if is_return_call(tc.name):
            return tc.input
    return None


def _user_prompt(prompt: str, context: Any) -> str:
    if context is None or context == "":
        return prompt
    return f"{prompt}\n\nContext:\n{json.dumps(jsonable(context), indent=2, default=str)}"


__all__ = [
    "DEFAULT_MAX_SCHEMA_RETRIES",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_SYSTEM_PROMPT",
    "run_step",
]
