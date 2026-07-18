"""Tests for insights.root_cause: regression judge narrative."""

from __future__ import annotations

import asyncio

import pytest

from ai_eval.insights.root_cause import explain


def _run(coro):
    return asyncio.run(coro)


def _fake_complete(rationale: str = "the model omitted a required step"):
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": 0.0, "rationale": rationale, "sub_scores": {}}
        )

    return complete_fn


def test_explain_returns_narrative() -> None:
    out = _run(
        explain(
            metric_name="faithfulness",
            current_output={"answer": "x"},
            baseline_output={"answer": "y"},
            current_score=0.4,
            baseline_score=0.9,
            example={"input": {"query": "q"}},
            regression_judge_model="fake/reg",
            complete_fn=_fake_complete("the answer dropped a citation"),
        )
    )
    assert "citation" in out


def test_explain_fallback_when_no_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the default transport to raise so the fallback narrative is used.
    async def boom(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("no transport")

    import ai_eval.judge.gateway as gw

    monkeypatch.setattr(gw, "_default_complete", boom)
    out = _run(
        explain(
            metric_name="faithfulness",
            current_output="a",
            baseline_output="b",
            current_score=0.4,
            baseline_score=0.9,
            example={},
            regression_judge_model="fake/reg",
            complete_fn=None,
        )
    )
    assert "regressed" in out
    assert "0.9" in out or "0.900" in out


def test_explain_fallback_on_judge_error() -> None:
    async def boom(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("judge down")

    out = _run(
        explain(
            metric_name="hallucination_rate",
            current_output="a",
            baseline_output="b",
            current_score=0.3,
            baseline_score=0.8,
            example={},
            regression_judge_model="fake/reg",
            complete_fn=boom,
        )
    )
    assert "regressed" in out
    assert "judge" in out.lower() or "unavailable" in out.lower()


def test_explain_no_regression_returns_fallback_without_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("no transport")

    import ai_eval.judge.gateway as gw

    monkeypatch.setattr(gw, "_default_complete", boom)
    out = _run(
        explain(
            metric_name="m",
            current_output="a",
            baseline_output="b",
            current_score=None,
            baseline_score=None,
            example={},
            regression_judge_model="fake/reg",
            complete_fn=None,
        )
    )
    # delta is None -> no "+/-0.xxx" segment but still names the metric
    assert "m" in out
    assert "regressed" in out
