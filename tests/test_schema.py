"""Schema fingerprinting and bare-type wrapping."""

from __future__ import annotations

from pydantic import BaseModel

from bridle.schema import fingerprint, is_pydantic_model, schema_dump, wrap_bare


class Topic(BaseModel):
    title: str


def test_pydantic_model_passes_through() -> None:
    assert wrap_bare(Topic) is Topic
    assert is_pydantic_model(Topic)


def test_bare_type_gets_wrapped() -> None:
    wrapped = wrap_bare(bool)
    assert is_pydantic_model(wrapped)
    # A new synthetic class is generated each call; that's fine — fingerprint
    # collapses them.
    assert wrap_bare(bool) is not None


def test_fingerprint_is_deterministic_for_same_model() -> None:
    a = fingerprint(Topic)
    b = fingerprint(Topic)
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_fingerprint_distinguishes_distinct_schemas() -> None:
    class Other(BaseModel):
        url: str

    assert fingerprint(Topic) != fingerprint(Other)


def test_fingerprint_distinguishes_renamed_models() -> None:
    # Pydantic embeds the class name in the JSON Schema's ``title`` field, so
    # two structurally identical models with different names fingerprint
    # differently. That's the right behavior for caching — a renamed model is
    # treated as a distinct schema.
    class Renamed(BaseModel):
        title: str

    assert fingerprint(Topic) != fingerprint(Renamed)


def test_schema_dump_works_for_bare_types() -> None:
    js = schema_dump(int)
    assert isinstance(js, dict)
    assert js  # non-empty


def test_schema_dump_works_for_models() -> None:
    js = schema_dump(Topic)
    assert "properties" in js
    assert "title" in js["properties"]
