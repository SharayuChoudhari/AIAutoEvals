"""Threshold evaluation + the run-time strict metric gate.

The :class:`~ai_eval.config.schema.MetricSpec` schema validator only *warns* on
unknown metrics (one-release deprecation window) unless
``AI_EVAL_STRICT_METRICS=1``. ``init``/``analyze`` stay lenient; ``run``
enforces strictly here: an unregistered metric raises
:class:`MetricNotImplementedError` (exit 1) with a fix hint, even when init
only warned. This keeps existing rubrics.yaml files loadable while making
``run`` fail fast on metrics it can't compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ai_eval.metrics.registry import get_judge_metric

ThresholdStatus = Literal["pass", "fail", "skip", "error"]


class MetricNotImplementedError(Exception):
    """Raised at run time when a rubrics.yaml metric has no implementation."""

    def __init__(self, name: str) -> None:
        super().__init__(f"metric {name!r} has no judge implementation; `run` cannot compute it")
        self.name = name
        self.why = (
            "the metric is not in the built-in judge set, not in the project's "
            "eval/metrics.yaml, and no plugin registered it"
        )
        self.fix = (
            f"add {name!r} to eval/metrics.yaml with a prompt_template (see "
            f"docs/metrics.md for a template), register it via the "
            f"'ai_eval.metrics' entry-point group, or remove it from rubrics.yaml"
        )


@dataclass
class ThresholdEval:
    score: float | None
    delta: float | None
    threshold: float
    status: ThresholdStatus  # pass | fail | skip | error


def evaluate_metric(
    name: str,
    score: float | None,
    *,
    threshold: float = 0.0,
    baseline_score: float | None = None,
    tolerance: float = 0.02,
    fail_on_regression: bool = False,
) -> ThresholdEval:
    """Evaluate one metric's score against its threshold + baseline delta.

    Fresh-repo (``baseline_score=None``): ``delta=None``, status from threshold
    only. ``--fail-on-regression`` only triggers on threshold breach, never on
    a negative delta alone (per locked decision #6).
    """
    delta: float | None = None
    if baseline_score is not None and score is not None:
        delta = score - baseline_score

    if score is None:
        return ThresholdEval(score=None, delta=delta, threshold=threshold, status="skip")

    # Threshold gate: score must be >= threshold.
    threshold_ok = score >= threshold

    # Regression gate: only under --fail-on-regression, and only when the
    # delta dropped below -tolerance. A bare delta without the flag is info-only.
    regressed = False
    if fail_on_regression and delta is not None and delta < -tolerance:
        regressed = True

    status: ThresholdStatus = "fail" if (not threshold_ok or regressed) else "pass"
    return ThresholdEval(score=score, delta=delta, threshold=threshold, status=status)


def assert_metric_implemented(name: str, project_root: Path | None = None) -> Any:
    """Run-time strict gate. Raises :class:`MetricNotImplementedError` if the
    metric has no judge implementation.

    ``project_root`` enables resolution of project-local ``eval/metrics.yaml``
    metrics; ``None`` falls back to built-ins + plugins only.
    """
    m = get_judge_metric(name, project_root=project_root)
    if m is None:
        raise MetricNotImplementedError(name)
    return m


__all__ = [
    "MetricNotImplementedError",
    "ThresholdEval",
    "assert_metric_implemented",
    "evaluate_metric",
]
