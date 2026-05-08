"""Trace events, observers, and serialization."""

from __future__ import annotations

import json

from bridle.trace import Event, Trace


def test_event_new_assigns_id_and_timestamp() -> None:
    event = Event.new("call_start", call_kind="step", label="x")
    assert event.id
    assert event.timestamp > 0
    assert event.kind == "call_start"
    assert event.call_kind == "step"
    assert event.label == "x"


def test_trace_appends_in_order() -> None:
    trace = Trace()
    a = Event.new("call_start", call_kind="step")
    b = Event.new("call_end", call_kind="step")
    trace.emit(a)
    trace.emit(b)
    assert [e.id for e in trace.events] == [a.id, b.id]
    assert len(trace) == 2
    assert list(iter(trace)) == [a, b]


def test_subscribe_fires_on_emit() -> None:
    trace = Trace()
    received: list[Event] = []
    trace.subscribe(received.append)
    event = Event.new("call_start", call_kind="step")
    trace.emit(event)
    assert received == [event]


def test_unsubscribe_stops_delivery() -> None:
    trace = Trace()
    received: list[Event] = []
    unsubscribe = trace.subscribe(received.append)
    trace.emit(Event.new("call_start"))
    unsubscribe()
    trace.emit(Event.new("call_end"))
    assert len(received) == 1


def test_to_dict_round_trips_basic_shape() -> None:
    trace = Trace()
    trace.emit(Event.new("call_start", call_kind="step", label="root"))
    rows = trace.to_dict()
    assert rows[0]["kind"] == "call_start"
    assert rows[0]["call_kind"] == "step"
    assert rows[0]["label"] == "root"


def test_to_jsonl_is_valid_jsonl() -> None:
    trace = Trace()
    trace.emit(Event.new("call_start", call_kind="step"))
    trace.emit(Event.new("call_end", call_kind="step", duration_ms=12.3))
    text = trace.to_jsonl()
    lines = text.split("\n")
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert "kind" in parsed
        assert "id" in parsed


def test_tree_groups_children_under_parents() -> None:
    trace = Trace()
    parent = Event.new("call_start", call_kind="agent")
    trace.emit(parent)
    child = Event.new("call_start", call_kind="step", parent_id=parent.id)
    trace.emit(child)
    grandchild = Event.new("model_request", parent_id=child.id)
    trace.emit(grandchild)

    tree = trace.tree()
    assert len(tree) == 1
    assert tree[0]["event"]["id"] == parent.id
    assert len(tree[0]["children"]) == 1
    assert tree[0]["children"][0]["event"]["id"] == child.id
    assert tree[0]["children"][0]["children"][0]["event"]["id"] == grandchild.id
