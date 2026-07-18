"""Tests for insights.diff: delta computation."""

from __future__ import annotations

from ai_eval.insights.diff import compute_delta


def _run_record(tasks):
    return {"tasks": tasks}


def test_compute_delta_basic() -> None:
    current = _run_record(
        {"t": {"metrics": {"m1": {"score": 0.8, "threshold": 0.5, "status": "pass"}}}}
    )
    baseline = _run_record(
        {"t": {"metrics": {"m1": {"score": 0.9, "threshold": 0.5, "status": "pass"}}}}
    )
    deltas = compute_delta(current, baseline)
    row = deltas["t"]["m1"]
    assert row["delta"] is not None
    assert abs(row["delta"] - (-0.1)) < 1e-9
    assert row["baseline_score"] == 0.9
    assert row["score"] == 0.8


def test_compute_delta_no_baseline() -> None:
    current = _run_record({"t": {"metrics": {"m1": {"score": 0.8}}}})
    deltas = compute_delta(current, None)
    assert deltas["t"]["m1"]["delta"] is None
    assert deltas["t"]["m1"]["baseline_score"] is None


def test_compute_delta_metric_absent_in_baseline() -> None:
    current = _run_record({"t": {"metrics": {"new": {"score": 0.7}}}})
    baseline = _run_record({"t": {"metrics": {}}})
    deltas = compute_delta(current, baseline)
    assert deltas["t"]["new"]["delta"] is None


def test_compute_delta_missing_score() -> None:
    current = _run_record({"t": {"metrics": {"m": {"score": None}}}})
    baseline = _run_record({"t": {"metrics": {"m": {"score": 0.9}}}})
    deltas = compute_delta(current, baseline)
    assert deltas["t"]["m"]["delta"] is None


def test_compute_delta_multiple_tasks() -> None:
    current = _run_record(
        {
            "t1": {"metrics": {"m": {"score": 0.5}}},
            "t2": {"metrics": {"m": {"score": 0.6}}},
        }
    )
    baseline = _run_record(
        {
            "t1": {"metrics": {"m": {"score": 0.5}}},
            "t2": {"metrics": {"m": {"score": 0.6}}},
        }
    )
    deltas = compute_delta(current, baseline)
    assert deltas["t1"]["m"]["delta"] == 0.0
    assert deltas["t2"]["m"]["delta"] == 0.0
