"""``brief_writer`` — the canonical Bridle example.

A research-brief assistant that exercises every primitive (``step``,
``branch``, ``loop``) and a couple of wrappers (``cache``, ``retry``).
Run with::

    ANTHROPIC_API_KEY=... python examples/brief_writer.py

The ``search`` tool is a stub — it returns plausible URLs without actually
hitting the web. Replace its body with a real search call to make the
output meaningful.
"""

from __future__ import annotations

import argparse

from pydantic import BaseModel

import bridle
from bridle import agent, branch, cache, loop, retry, step, tool
from bridle.models.anthropic import install


class Topic(BaseModel):
    title: str
    angle: str


class Plan(BaseModel):
    topics: list[Topic]


class Source(BaseModel):
    url: str
    summary: str


class Brief(BaseModel):
    headline: str
    body: str


@tool
def search(query: str) -> list[str]:
    """Search the web. Returns up to 10 result URLs.

    This is a stub for the example — replace with a real search backend.
    """

    slug = query.lower().replace(" ", "-")[:40]
    return [f"https://example.com/{slug}/{i}" for i in range(5)]


@agent(input=str, output=Brief, model="claude-sonnet-4-6", token_budget=200_000)
def brief_writer(topic: str) -> Brief:
    plan = cache(step("draft a research plan with two distinct angles", schema=Plan, context=topic))

    sources: list[Source] = []
    for t in plan.topics:
        found = loop(
            f"gather distinct sources on {t.title}",
            schema=Source,
            until=lambda acc: len(acc) >= 2,
            tools=[search],
            max_iterations=4,
        )
        sources.extend(found)

    if not branch("is the evidence sufficient?", context=sources):
        return brief_writer(f"{topic} — go deeper on whatever's underdocumented")

    return retry(
        step("write the brief", schema=Brief, context=(topic, sources)),
        attempts=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a research brief on a topic.")
    parser.add_argument("topic", nargs="?", default="the weather on Mars")
    args = parser.parse_args()

    install()  # register the Anthropic adapter as the active model client

    brief = bridle.resolve(brief_writer(args.topic))

    print(f"# {brief.headline}\n")
    print(brief.body)


if __name__ == "__main__":
    main()
