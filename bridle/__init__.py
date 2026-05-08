"""Bridle — harness an agent into a deterministic program with typed I/O."""

from __future__ import annotations

from .call import Call, CallMeta, evaluate, register, resolve, unregister
from .errors import (
    BridleError,
    ConfigurationError,
    HumanAbortedError,
    LoopExhaustedError,
    ModelError,
    SchemaSatisfactionError,
    TimeoutError,
    TokenBudgetExceededError,
    ToolExecutionError,
)
from .primitives import step
from .runtime import (
    configure,
    current_cache,
    current_human_channel,
    current_model,
    current_token_budget,
    set_cache,
    set_human_channel,
)
from .tool import Tool
from .trace import Event, Trace, current_trace

__version__ = "0.1.0"

__all__ = [
    "BridleError",
    "Call",
    "CallMeta",
    "ConfigurationError",
    "Event",
    "HumanAbortedError",
    "LoopExhaustedError",
    "ModelError",
    "SchemaSatisfactionError",
    "TimeoutError",
    "TokenBudgetExceededError",
    "Tool",
    "ToolExecutionError",
    "Trace",
    "__version__",
    "configure",
    "current_cache",
    "current_human_channel",
    "current_model",
    "current_token_budget",
    "current_trace",
    "evaluate",
    "register",
    "resolve",
    "set_cache",
    "set_human_channel",
    "step",
    "unregister",
]
