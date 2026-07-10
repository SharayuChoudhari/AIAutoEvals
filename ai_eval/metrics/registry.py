"""Extensible metric registry.

Built-in metrics ship in code (:data:`BUILTIN_METRICS`). Third parties extend
the set via the ``ai_eval.metrics`` entry-point group, which is loaded lazily by
:func:`load_metrics` (mirroring ``load_entrypoint_detectors`` in
``ast_scan.py:79``).

The :class:`~ai_eval.config.schema.MetricSpec` schema validator consults
:func:`is_registered` to reject unknown metric names, keeping both the rule and
SLM engines on the same validation surface.

A one-release deprecation window: unknown names emit a warning via
:func:`warn_unknown` rather than being rejected outright. Set the env var
``AI_EVAL_STRICT_METRICS=1`` to enforce hard rejection now.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Metric:
    """Describes one metric the SLM/rule engine is allowed to emit."""

    name: str
    description: str
    default_threshold: float = 0.0
    default_weight: float = 1.0
    applicable_task_types: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class JudgeMetric:
    """A judge-driven metric: a prompt builder plus a canonical sub-score.

    ``prompt_builder`` receives ``(task_spec, example, output)`` and returns a
    chat ``messages`` list. ``scored_dimension`` names the
    :class:`~ai_eval.judge.schemas.JudgeResponse` sub-score the runner lifts
    into the metric value. ``non_judge=True`` marks latency-style metrics that
    the runner computes locally (no judge call); their ``prompt_builder`` is
    unused.
    """

    name: str
    description: str
    applicable_task_types: tuple[str, ...] = field(default_factory=tuple)
    scored_dimension: str = "score"
    prompt_builder: Callable[..., list[dict[str, str]]] | None = None
    default_threshold: float = 0.0
    default_weight: float = 1.0
    non_judge: bool = False


BUILTIN_METRICS: tuple[Metric, ...] = (
    Metric(
        name="hallucination_rate",
        description="Rate of unsupported claims in the output relative to grounded context.",
        default_threshold=0.1,
        applicable_task_types=("chat", "rag", "agent", "tool_calling"),
    ),
    Metric(
        name="context_precision",
        description="Precision of retrieved context relevance to the query.",
        default_threshold=0.85,
        applicable_task_types=("rag",),
    ),
    Metric(
        name="faithfulness",
        description="Faithfulness of the answer to the retrieved context.",
        default_threshold=0.9,
        applicable_task_types=("rag", "summarization"),
    ),
    Metric(
        name="argument_accuracy",
        description="Accuracy of tool/function call arguments vs. expected.",
        default_threshold=0.9,
        applicable_task_types=("tool_calling", "agent"),
    ),
    Metric(
        name="scoring_accuracy",
        description="Agreement of model scores with reference rubric scores.",
        default_threshold=0.85,
        applicable_task_types=("scoring",),
    ),
    Metric(
        name="extraction_field_accuracy",
        description="Per-field accuracy of extracted structured fields vs. reference.",
        default_threshold=0.9,
        applicable_task_types=("extraction",),
    ),
    Metric(
        name="classification_f1",
        description="Macro-F1 across classification labels.",
        default_threshold=0.85,
        applicable_task_types=("classification",),
    ),
    Metric(
        name="summary_faithfulness",
        description="Faithfulness of a summary to its source text.",
        default_threshold=0.9,
        applicable_task_types=("summarization",),
    ),
    Metric(
        name="translation_bleu",
        description="BLEU score of translated output against references.",
        default_threshold=0.4,
        applicable_task_types=("translation",),
    ),
    Metric(
        name="task_completion",
        description="Whether the end-to-end task (e.g. booking, agent) completed successfully.",
        default_threshold=0.9,
        applicable_task_types=("booking", "workflow", "agent"),
    ),
    Metric(
        name="slot_filling_accuracy",
        description="Accuracy of slots/arguments filled for booking or workflow tasks.",
        default_threshold=0.9,
        applicable_task_types=("booking", "workflow"),
    ),
)

def _latency_p50(_: Any = None, __: Any = None, ___: Any = None) -> list[dict[str, str]]:
    return []  # pragma: no cover - non_judge; never called


def _latency_p95(_: Any = None, __: Any = None, ___: Any = None) -> list[dict[str, str]]:
    return []  # pragma: no cover - non_judge; never called


BUILTIN_JUDGE_METRICS_TUPLE: tuple[JudgeMetric, ...] = (
    JudgeMetric(
        name="latency_p50",
        description=(
            "Median (p50) task latency in milliseconds. "
            "Computed by the runner; no judge call."
        ),
        applicable_task_types=(),
        scored_dimension="latency_ms",
        prompt_builder=_latency_p50,
        non_judge=True,
    ),
    JudgeMetric(
        name="latency_p95",
        description="p95 task latency in milliseconds. Computed by the runner; no judge call.",
        applicable_task_types=(),
        scored_dimension="latency_ms",
        prompt_builder=_latency_p95,
        non_judge=True,
    ),
)

_EP_GROUP = "ai_eval.metrics"

#: Memoized metric set. Entry-point discovery + plugin loading is relatively
#: expensive and runs on every ``is_registered``/``get`` call (the schema
#: validator calls ``is_registered`` per metric name), so we cache the result
#: for the process. ``reset_cache`` (called by tests that patch entry points)
#: invalidates it.
_metrics_cache: tuple[Metric, ...] | None = None
_judge_metrics_cache: tuple[JudgeMetric, ...] | None = None


def reset_cache() -> None:
    """Invalidate the memoized metric sets (used by tests that patch plugins)."""
    global _metrics_cache, _judge_metrics_cache
    _metrics_cache = None
    _judge_metrics_cache = None


def load_metrics() -> tuple[Metric, ...]:
    """Return built-in metrics plus any contributed via the entry-point group.

    Memoized for the process: entry-point discovery and plugin loading happen
    once, since the registry is consulted on every ``MetricSpec`` validation.
    """
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache
    builtins = BUILTIN_METRICS
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        _metrics_cache = builtins
        return builtins
    extra: list[Metric] = []
    try:
        eps = entry_points(group=_EP_GROUP)
    except TypeError:
        eps = entry_points().get(_EP_GROUP, [])  # type: ignore[attr-defined]
    seen = {m.name for m in builtins}
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            continue
        metric = _coerce_plugin(obj, ep.name)
        if metric is None:
            continue
        if metric.name in seen:
            continue
        seen.add(metric.name)
        extra.append(metric)
    _metrics_cache = (*builtins, *extra)
    return _metrics_cache


def load_judge_metrics() -> tuple[JudgeMetric, ...]:
    """Return built-in judge metrics plus plugin-contributed ones.

    Built-ins come from ``ai_eval.metrics.judge_builtin`` (the four judge
    metrics) plus the local latency pair. Plugins expose a ``JudgeMetric``
    (or a compatible dataclass/dict) under the same ``ai_eval.metrics`` group.
    """
    global _judge_metrics_cache
    if _judge_metrics_cache is not None:
        return _judge_metrics_cache
    from ai_eval.metrics.judge_builtin import BUILTIN_JUDGE_METRICS

    builtins: tuple[JudgeMetric, ...] = (
        *BUILTIN_JUDGE_METRICS,
        *BUILTIN_JUDGE_METRICS_TUPLE,
    )
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        _judge_metrics_cache = builtins
        return builtins
    extra: list[JudgeMetric] = []
    try:
        eps = entry_points(group=_EP_GROUP)
    except TypeError:
        eps = entry_points().get(_EP_GROUP, [])  # type: ignore[attr-defined]
    seen = {m.name for m in builtins}
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:
            continue
        metric = _coerce_judge_plugin(obj, ep.name)
        if metric is None or metric.name in seen:
            continue
        seen.add(metric.name)
        extra.append(metric)
    _judge_metrics_cache = (*builtins, *extra)
    return _judge_metrics_cache


def _coerce_judge_plugin(obj: object, ep_name: str) -> JudgeMetric | None:
    if isinstance(obj, JudgeMetric):
        return obj
    if isinstance(obj, dict):
        try:
            return JudgeMetric(
                name=str(obj["name"]),
                description=str(obj.get("description", "")),
                applicable_task_types=tuple(obj.get("applicable_task_types", []) or ()),
                scored_dimension=str(obj.get("scored_dimension", "score")),
                prompt_builder=obj.get("prompt_builder"),
                non_judge=bool(obj.get("non_judge", False)),
                default_threshold=float(obj.get("default_threshold", 0.0)),
                default_weight=float(obj.get("default_weight", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(obj, str):
        return JudgeMetric(
            name=obj, description=f"plugin judge metric {ep_name}", non_judge=False
        )
    return None


def _coerce_plugin(obj: object, ep_name: str) -> Metric | None:
    if isinstance(obj, Metric):
        return obj
    if isinstance(obj, dict):
        try:
            return Metric(
                name=str(obj["name"]),
                description=str(obj.get("description", "")),
                default_threshold=float(obj.get("default_threshold", 0.0)),
                default_weight=float(obj.get("default_weight", 1.0)),
                applicable_task_types=tuple(obj.get("applicable_task_types", []) or ()),
            )
        except (KeyError, TypeError, ValueError):
            return None
    # A bare string entry-point value means "register this name with no metadata".
    if isinstance(obj, str):
        return Metric(name=obj, description=f"plugin metric {ep_name}")
    return None


def all_names() -> list[str]:
    """All registered metric names (legacy + judge + latency)."""
    names = {m.name for m in load_metrics()}
    names |= {m.name for m in load_judge_metrics()}
    return sorted(names)


def is_registered(name: str) -> bool:
    if name in {m.name for m in load_metrics()}:
        return True
    return name in {m.name for m in load_judge_metrics()}


def get(name: str) -> Metric | None:
    for m in load_metrics():
        if m.name == name:
            return m
    return None


def get_judge_metric(name: str) -> JudgeMetric | None:
    """Return the judge metric for ``name`` or ``None`` if not registered.

    The run-time strict gate (:mod:`ai_eval.runner.thresholds`) calls this and
    raises :class:`MetricNotImplementedError` (exit 1) when a rubrics.yaml
    metric has no implementation — even when init-time validation only warned.
    """
    for m in load_judge_metrics():
        if m.name == name:
            return m
    return None


def is_strict() -> bool:
    return os.environ.get("AI_EVAL_STRICT_METRICS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def warn_unknown(name: str) -> None:
    """One-release deprecation path: warn (not reject) unknown metric names.

    When ``AI_EVAL_STRICT_METRICS`` is truthy this is a no-op and the caller
    should reject. Otherwise a ``DeprecationWarning`` is emitted so existing
    rubrics.yaml files survive until the next major release.
    """
    if is_strict():
        return
    warnings.warn(
        f"metric {name!r} is not in the metric registry; it will be rejected in a "
        f"future release. Register it via the 'ai_eval.metrics' entry-point group, "
        f"or set AI_EVAL_STRICT_METRICS=1 to enforce now.",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = [
    "BUILTIN_JUDGE_METRICS_TUPLE",
    "BUILTIN_METRICS",
    "JudgeMetric",
    "Metric",
    "all_names",
    "get",
    "get_judge_metric",
    "is_registered",
    "is_strict",
    "load_judge_metrics",
    "load_metrics",
    "reset_cache",
    "warn_unknown",
]
