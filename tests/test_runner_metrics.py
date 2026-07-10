"""Tests for latency metric computation."""

from __future__ import annotations

from ai_eval.runner.metrics.latency import compute_latencies, percentile


def test_percentile_single_value() -> None:
    assert percentile([5.0], 50) == 5.0
    assert percentile([5.0], 95) == 5.0


def test_percentile_basic() -> None:
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0
    assert percentile(vals, 50) == 3.0


def test_compute_latencies_empty() -> None:
    out = compute_latencies([])
    assert out == {"p50": 0.0, "p95": 0.0}


def test_compute_latencies_nonempty() -> None:
    out = compute_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert out["p50"] == 30.0
    assert out["p95"] >= 40.0


def test_compute_latencies_all_equal() -> None:
    out = compute_latencies([7.0, 7.0, 7.0])
    assert out["p50"] == 7.0
    assert out["p95"] == 7.0
