# Bridle

Harness an agent into a deterministic program with typed I/O.

Bridle is a Python library that lets you write agentic code the way you'd write any program: control flow in code, judgment in the model. Five primitives are typed holes the model fills with values — `step`, `branch`, `loop`, `ask_human`, `parallel`. Two decorators compose them: `@agent`, `@tool`. Wrappers add behavior: `cache`, `retry`, `timeout`, `with_model`, `fallback`, `mock`, `log`.

It reads like async/await for LLM decisions.

## Status

v0.1.0 — Anthropic-only, sync, single-agent.

## Install

```bash
pip install bridle-ai
```

The PyPI distribution is `bridle-ai`; the import is `bridle`.

## Quickstart

```python
from bridle import agent, step, branch, loop, ask_human, parallel, tool, cache, retry
from pydantic import BaseModel

class Topic(BaseModel): title: str
class Plan(BaseModel): topics: list[Topic]
class Source(BaseModel): url: str; summary: str
class Brief(BaseModel): headline: str; body: str

@tool
def search(query: str) -> list[str]:
    """Web search. Returns up to 10 result URLs."""
    ...

@agent(input=str, output=Brief, model="claude-sonnet-4-6")
def brief_writer(topic: str) -> Brief:
    plan = cache(step("draft a research plan", schema=Plan, context=topic))

    gathered = parallel(*[
        loop(
            f"gather sources on {t.title}",
            schema=Source,
            until=lambda acc: len(acc) >= 3,
            tools=[search],
        )
        for t in plan.topics
    ])
    sources = [s for batch in gathered for s in batch]

    if not branch("is the evidence sufficient?", context=sources):
        guidance = ask_human("what angle is missing?")
        return brief_writer(f"{topic} — emphasis: {guidance}")

    return retry(step("write the brief", schema=Brief, context=(topic, sources)), attempts=2)
```

## Model selection

Bridle ships zero default models. Set one of three places:

```python
import bridle
bridle.configure(model="claude-sonnet-4-6")        # process-wide
```

```python
@agent(model="claude-opus-4-7", ...)               # per-agent
def my_agent(...): ...
```

```python
with_model(step("..."), "claude-haiku-4-5")        # per-call
```

Resolution: per-call → per-agent → process. If none is set, Bridle raises `ConfigurationError`.

## What's next (v2)

- Async-first execution
- Multi-agent coordination
- Streaming primitives
- Redis-backed cache and durability
- Model abstraction beyond Anthropic
- Sealed inner traces (`seal=True` on `@agent`)

## License

MIT.
