"""Bridle — harness an agent into a deterministic program with typed I/O."""

from __future__ import annotations

from .agent import agent
from .call import Call, CallMeta, evaluate, register, resolve, unregister
from .errors import (
    BridleError,
    ConfigurationError,
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
    current_model,
    current_token_budget,
    current_token_usage,
    set_cache,
)
from .tool import Tool, tool
from .trace import Event, Trace, current_trace

__version__ = "0.1.0"

__all__ = [
    "BridleError",
    "Call",
    "CallMeta",
    "ConfigurationError",
    "Event",
    "LoopExhaustedError",
    "ModelError",
    "SchemaSatisfactionError",
    "TimeoutError",
    "TokenBudgetExceededError",
    "Tool",
    "ToolExecutionError",
    "Trace",
    "__version__",
    "agent",
    "configure",
    "current_cache",
    "current_model",
    "current_token_budget",
    "current_token_usage",
    "current_trace",
    "evaluate",
    "register",
    "resolve",
    "set_cache",
    "step",
    "tool",
    "unregister",
]
