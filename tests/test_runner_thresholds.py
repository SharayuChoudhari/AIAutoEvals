"""Tests for the runner thresholds module + run-time strict gate."""

from __future__ import annotations

import pytest

from ai_eval.metrics import reset_cache
from ai_eval.runner.thresholds import (
    MetricNotImplementedError,
    assert_metric_implemented,
    evaluate_metric,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_cache()
    yield
    reset_cache()


def test_evaluate_pass_when_above_threshold() -> None:
    ev = evaluate_metric("m", 0.95, threshold=0.9)
    assert ev.status == "pass"
    assert ev.score == 0.95


def test_evaluate_fail_when_below_threshold() -> None:
    ev = evaluate_metric("m", 0.8, threshold=0.9)
    assert ev.status == "fail"


def test_evaluate_skip_when_score_none() -> None:
    ev = evaluate_metric("m", None, threshold=0.9)
    assert ev.status == "skip"
    assert ev.delta is None


def test_delta_computed_when_baseline_present() -> None:
    ev = evaluate_metric("m", 0.85, threshold=0.8, baseline_score=0.9)
    assert ev.delta == pytest.approx(-0.05)


def test_delta_none_when_no_baseline() -> None:
    ev = evaluate_metric("m", 0.85, threshold=0.8, baseline_score=None)
    assert ev.delta is None


def test_fail_on_regression_triggers_on_drop_beyond_tolerance() -> None:
    ev = evaluate_metric(
        "m", 0.85, threshold=0.0, baseline_score=0.95,
        tolerance=0.02, fail_on_regression=True,
    )
    assert ev.status == "fail"  # dropped 0.10 > tolerance 0.02


def test_fail_on_regression_no_trigger_within_tolerance() -> None:
    ev = evaluate_metric(
        "m", 0.89, threshold=0.0, baseline_score=0.9,
        tolerance=0.02, fail_on_regression=True,
    )
    assert ev.status == "pass"


def test_delta_without_fail_flag_is_info_only() -> None:
    # regression but no flag -> status from threshold only
    ev = evaluate_metric("m", 0.5, threshold=0.0, baseline_score=0.9)
    assert ev.status == "pass"  # above threshold, delta ignored


def test_assert_implemented_for_builtin() -> None:
    m = assert_metric_implemented("faithfulness")
    assert m is not None
    assert m.scored_dimension == "faithfulness"


def test_assert_implemented_raises_for_unknown() -> None:
    with pytest.raises(MetricNotImplementedError) as excinfo:
        assert_metric_implemented("no_such_metric_xyz")
    assert "no_such_metric_xyz" in str(excinfo.value)
    assert excinfo.value.fix


def test_latency_metrics_are_implemented() -> None:
    assert assert_metric_implemented("latency_p50") is not None
    assert assert_metric_implemented("latency_p95") is not None
