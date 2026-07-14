"""Tests for built-in judge metrics: prompt builder output + scored_dimension."""

from __future__ import annotations

import json

import pytest

from ai_eval.metrics import get_judge_metric, load_judge_metrics, reset_cache
from ai_eval.metrics.judge_builtin import BUILTIN_JUDGE_METRICS


@pytest.fixture(autouse=True)
def _reset():
    reset_cache()
    yield
    reset_cache()


_NAMES = {"argument_accuracy", "hallucination_rate", "context_precision", "faithfulness"}


def test_builtin_judge_metrics_present() -> None:
    names = {m.name for m in BUILTIN_JUDGE_METRICS}
    assert _NAMES <= names


def test_registry_exposes_judge_metrics() -> None:
    all_names = {m.name for m in load_judge_metrics()}
    assert _NAMES <= all_names
    assert {"latency_p50", "latency_p95"} <= all_names


def test_get_judge_metric_returns_scored_dimension() -> None:
    m = get_judge_metric("faithfulness")
    assert m is not None
    assert m.scored_dimension == "faithfulness"


def test_latency_metrics_are_non_judge() -> None:
    p50 = get_judge_metric("latency_p50")
    p95 = get_judge_metric("latency_p95")
    assert p50 is not None and p50.non_judge
    assert p95 is not None and p95.non_judge


def test_get_judge_metric_unknown_returns_none() -> None:
    assert get_judge_metric("does_not_exist") is None


def _task(type_="rag", purpose="answer the query"):
    class _T:
        pass

    t = _T()
    t.type = type_
    t.purpose = purpose
    return t


def test_argument_accuracy_prompt_shape() -> None:
    m = get_judge_metric("argument_accuracy")
    assert m is not None and m.prompt_builder is not None
    msgs = m.prompt_builder(_task("tool_calling"), {"input": {"q": 1}}, {"arg": "x"})
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "argument_accuracy" in json.dumps(msgs)


def test_hallucination_prompt_uses_checklist_for_basic_tier() -> None:
    m = get_judge_metric("hallucination_rate")
    msgs = m.prompt_builder(_task("chat"), {"input": {}}, "out", tier="basic")
    content = json.dumps(msgs)
    assert "yes" in content.lower() or "no" in content.lower()


def test_context_precision_includes_deterministic_signal() -> None:
    m = get_judge_metric("context_precision")
    example = {
        "input": {"query": "what is the return policy"},
        "trace": {
            "calls": [
                {"kind": "retrieve", "result": "return policy allows 30 days"},
                {"kind": "retrieve", "result": "weather in london"},
            ]
        },
    }
    msgs = m.prompt_builder(_task("rag"), example, "out")
    content = json.dumps(msgs)
    # deterministic signal computed from query/term overlap
    assert "deterministic" in content.lower() or "signal" in content.lower()


def test_faithfulness_prompt_complex_tier() -> None:
    m = get_judge_metric("faithfulness")
    msgs = m.prompt_builder(_task("rag"), {"input": {"query": "q"}}, "answer")
    assert msgs[0]["role"] == "system"
    assert "faithfulness" in json.dumps(msgs)


def test_prompt_builder_handles_example_without_input_key() -> None:
    m = get_judge_metric("faithfulness")
    # example with no 'input' key — should still build
    msgs = m.prompt_builder(_task("rag"), {}, "answer")
    assert len(msgs) == 2
