# 🐴 Bridle

Harness an agent into a deterministic program with typed I/O.

Bridle is a Python library for writing agents the way you'd write any program: control flow in code, judgment in the model. Three primitives are typed holes the model fills with values — `step`, `branch`, `loop`. Two decorators compose them: `@agent`, `@tool`. Wrappers add behavior: `cache`, `retry`, `timeout`, `with_model`, `fallback`, `mock`, `log`.

It reads like `async`/`await` for LLM decisions.

## Status

v0.1.0 — Anthropic-only, sync, single-agent. See [What's next](#whats-next) for the v0.2.0 roadmap.

## Install

```bash
pip install bridle-ai
```

The PyPI distribution is `bridle-ai`; the import is `bridle`.

## Quickstart

```python
import bridle
from bridle import agent, branch, cache, loop, retry, step, tool
from bridle.models.anthropic import install
from pydantic import BaseModel


class Topic(BaseModel): title: str
class Plan(BaseModel): topics: list[Topic]
class Source(BaseModel): url: str; summary: str
class Brief(BaseModel): headline: str; body: str


@tool
def search(query: str) -> list[str]:
    """Search the web. Returns up to 10 result URLs."""
    ...


@agent(input=str, output=Brief, model="claude-sonnet-4-6")
def brief_writer(topic: str) -> Brief:
    plan = cache(step("draft a research plan", schema=Plan, context=topic))

    sources: list[Source] = []
    for t in plan.topics:
        found = loop(
            f"gather sources on {t.title}",
            schema=Source,
            until=lambda acc: len(acc) >= 3,
            tools=[search],
        )
        sources.extend(found)

    if not branch("is the evidence sufficient?", context=sources):
        return brief_writer(f"{topic} — go deeper on whatever's underdocumented")

    return retry(step("write the brief", schema=Brief, context=(topic, sources)), attempts=2)


install()  # registers the Anthropic adapter as the active model client
result = bridle.resolve(brief_writer("the weather on Mars"))
print(result.headline)
```

The model never picks the next state. It produces typed values; your Python decides where to go next. Every primitive is mockable; every run is observable.

## The four primitives

| Primitive | What it does |
| --- | --- |
| `step(prompt, *, schema, context=None, tools=())` | The atomic unit: the model works toward a typed return, calling tools as needed, until its output satisfies the schema. |
| `branch(prompt, *, schema=bool, context=None)` | A step constrained to a single typed decision. Defaults to `bool`; pass an `Enum` or `Literal` for multi-way. |
| `loop(prompt, *, schema, until, tools=(), max_iterations=32)` | Repeat a step until a pure-Python predicate is satisfied. `LoopExhaustedError` on cap. |
| `@agent(input=, output=, model=, token_budget=)` | Wrap a Python function whose body uses primitives. Validates I/O. Inner steps inherit the agent's model. |

`@tool` registers a Python function as a tool the model can call. The parameter schema is extracted from type hints; the docstring becomes the description.

## The wrapper algebra

Every wrapper takes a `Call` and returns a `Call`. They compose freely.

```python
cache(retry(timeout(step("..."), seconds=10), attempts=3))
```

| Wrapper | What it does |
| --- | --- |
| `cache(call, *, key=None, backend=None, ttl=None)` | Memoize results. Default key hashes kind + schema + context + prompt + tools. |
| `retry(call, *, attempts=3, on=BridleError, backoff=None)` | Re-evaluate on failure. Each attempt clones the inner call. |
| `timeout(call, *, seconds)` | Abort if the call runs past the deadline. Raises `bridle.TimeoutError`. |
| `with_model(call, "model-id")` | Per-call model override (highest layer of model resolution). |
| `fallback(call, *alternates)` | Try each in turn until one succeeds. |
| `mock(call, value)` | Replace dispatch with a constant. For tests. |
| `log(call, *, level="INFO")` | Stream the trace to a Python logger for the wrapper's subtree. |

## Model selection

Bridle ships zero default models. Set one of three layers:

```python
bridle.configure(model="claude-sonnet-4-6")     # process-wide
@agent(model="claude-opus-4-7", ...)             # per-agent
with_model(step("..."), "claude-haiku-4-5")     # per-call (highest precedence)
```

Resolution order: per-call → per-agent → process. If none is set, Bridle raises `ConfigurationError` with a message that lists all three places.

## The trace

Every primitive emits structured events into a `Trace` you can inspect, replay, or stream.

```python
from bridle import Trace
from bridle.trace import set_active_trace

trace = Trace()
set_active_trace(trace)

bridle.resolve(brief_writer("..."))

print(trace.to_jsonl())          # one JSON line per event
print(trace.tree())              # nested view: agent → step → model_request → ...
trace.subscribe(lambda e: ...)   # live observer
```

Event kinds: `call_start`, `call_end`, `model_request`, `model_response`, `tool_call`, `tool_result`, `cache_hit`, `cache_miss`, `retry`.

## Token budgets

Soft per-agent budget — usage accumulates after each model response and the next step raises `TokenBudgetExceededError` if it would breach.

```python
@agent(input=Q, output=R, model="claude-sonnet-4-6", token_budget=100_000)
def deep_research(q: Q) -> R:
    ...
```

## Caching

Caching is opt-in per call via the `cache` wrapper. Backends ship for memory and file; Redis is reserved for v0.2.0.

```python
from bridle.cache.file import FileCache
bridle.set_cache(FileCache("./.bridle-cache"))

# Now any cache(...) wrapped call writes to disk.
plan = cache(step("draft a plan", schema=Plan, context=topic))
```

The default key is deterministic across runs: it hashes the call's kind, schema fingerprint, context, prompt, and tools.

## What's next

v0.2.0 (planned):
- Async-first execution
- `parallel` primitive (returning to the four primitives the brief originally proposed, native to async)
- Multi-agent coordination
- Streaming primitives
- Redis-backed cache
- Sealed inner traces (`seal=True` on `@agent`)
- Model abstraction beyond Anthropic
- True durable execution for long-running agents

## License

MIT.
