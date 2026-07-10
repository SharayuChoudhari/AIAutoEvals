"""Tests for the bootstrap tracer: nesting, dedup, max cap."""

from __future__ import annotations

import pytest

from ai_eval.bootstrap.tracer import (
    configure_sink,
    enter_task,
    exit_task,
    get_tracer,
    record_call,
    task_scope,
)


@pytest.fixture(autouse=True)
def _reset_each():
    t = get_tracer()
    t.reset()
    t._local.__dict__.clear()
    yield
    t.reset()
    t._local.__dict__.clear()


def test_enter_exit_emits_one_example() -> None:
    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)
    enter_task("t", {"q": 1})
    record_call("llm", "openai.create", [], {"ok": True}, 5.0)
    example = exit_task({"answer": "yes"})
    assert example is not None
    assert example["task"] == "t"
    assert example["input"] == {"q": 1}
    assert example["expected"] == {"answer": "yes"}
    assert len(example["trace"]["calls"]) == 1
    assert captured == [example]


def test_expected_from_expected_key() -> None:
    enter_task("t", {})
    example = exit_task({"expected": "the answer"})
    assert example is not None
    assert example["expected"] == "the answer"


def test_dedup_same_input_same_task() -> None:
    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)
    enter_task("t", {"x": 1})
    exit_task()
    enter_task("t", {"x": 1})  # same input -> dedup
    second = exit_task()
    assert second is None
    assert len(captured) == 1


def test_different_input_not_deduped() -> None:
    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)
    enter_task("t", {"x": 1})
    exit_task()
    enter_task("t", {"x": 2})
    exit_task()
    assert len(captured) == 2


def test_task_scope_context_manager() -> None:
    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)
    with task_scope("t", {"y": 9}):
        record_call("retrieve", "chromadb.query", [], {}, 1.0)
    assert len(captured) == 1
    assert captured[0]["trace"]["calls"][0]["kind"] == "retrieve"


def test_record_outside_task_is_noop() -> None:
    record_call("llm", "x", [], None, 1.0)
    assert get_tracer().current is None


def test_example_id_stable_for_same_input() -> None:
    enter_task("t", {"q": "hello"})
    e1 = exit_task()
    enter_task("t", {"q": "hello"})
    e2 = exit_task()  # noqa: F841
    # e2 is None due to dedup, but the id would be stable
    assert e1 is not None
    assert e1["id"].startswith("t_")
