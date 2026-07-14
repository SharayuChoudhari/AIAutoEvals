"""Extensible metric registry.

Built-in metrics ship in code (:data:`BUILTIN_METRICS`) — the high-frequency
set only. Niche/project-specific metrics live in the consuming project's
``eval/metrics.yaml``, loaded and merged at runtime by
:mod:`ai_eval.metrics.local`. Third parties also extend the set via the
``ai_eval.metrics`` entry-point group, loaded lazily by :func:`load_metrics`
(mirroring ``load_entrypoint_detectors`` in ``ast_scan.py:79``).

The :class:`~ai_eval.config.schema.MetricSpec` schema validator consults
:func:`is_registered` to reject unknown metric names, keeping both the rule and
SLM engines on the same validation surface.

A one-release deprecation window: unknown names emit a warning via
:func:`warn_unknown` rather than being rejected outright. Set the env var
``AI_EVAL_STRICT_METRICS=1`` to enforce hard rejection now.

Merge order in :func:`load_metrics` / :func:`load_judge_metrics`:
1. Built-ins (always win on name collision).
2. Project-local ``eval/metrics.yaml`` (can only ADD names, never override
   built-ins).
3. Entry-point plugins (dedup by ``seen`` set).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
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


#: The high-frequency built-in metric set. Niche/project-specific metrics
#: (slot_filling_accuracy, translation_bleu, classification_f1,
#: extraction_field_accuracy, scoring_accuracy, summary_faithfulness, and any
#: custom metric) live in the consuming project's ``eval/metrics.yaml`` — see
#: :mod:`ai_eval.metrics.local` and ``docs/metrics.md``. Do NOT add niche
#: metrics back here; built-ins stay the high-frequency set only.
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
        name="task_completion",
        description="Whether the end-to-end task (e.g. booking, agent) completed successfully.",
        default_threshold=0.9,
        applicable_task_types=("booking", "workflow", "agent"),
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
            "Median (p50) task latency in milliseconds. Computed by the runner; no judge call."
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

#: Memoized metric sets, keyed by ``project_root`` (or ``None`` when no root is
#: available, e.g. the schema validator). Entry-point discovery + plugin loading
#: is relatively expensive and runs on every ``is_registered``/``get`` call (the
#: schema validator calls ``is_registered`` per metric name), so we cache the
#: result per process-root. ``reset_cache`` (called by tests that patch entry
#: points or change the local metrics file) invalidates it.
_metrics_cache: dict[Path | None, tuple[Metric, ...]] = {}
_judge_metrics_cache: dict[Path | None, tuple[JudgeMetric, ...]] = {}


def reset_cache() -> None:
    """Invalidate the memoized metric sets (used by tests that patch plugins)."""
    _metrics_cache.clear()
    _judge_metrics_cache.clear()


def load_metrics(project_root: Path | None = None) -> tuple[Metric, ...]:
    """Return built-in + project-local + plugin metrics.

    Memoized for the process (keyed by ``project_root`` so a test that changes
    the root sees fresh local metrics): entry-point discovery and plugin loading
    happen once per root, since the registry is consulted on every
    ``MetricSpec`` validation.

    Merge order — built-ins always win on name collision; project-local
    ``eval/metrics.yaml`` can only ADD names; entry-point plugins fill the rest.
    """
    cache_key = project_root
    cached = _metrics_cache.get(cache_key)
    if cached is not None:
        return cached
    builtins = BUILTIN_METRICS
    local: tuple[Metric, ...] = ()
    if project_root is not None:
        from ai_eval.metrics.local import load_local_metrics

        local = load_local_metrics(project_root)
    extra: list[Metric] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        result = (*builtins, *local)
        _metrics_cache[cache_key] = result
        return result
    try:
        eps = entry_points(group=_EP_GROUP)
    except TypeError:
        eps = entry_points().get(_EP_GROUP, [])  # type: ignore[attr-defined]
    seen = {m.name for m in builtins}
    # Project-local can only ADD names (never override built-ins).
    for m in local:
        if m.name not in seen:
            seen.add(m.name)
            extra.append(m)
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
    result = (*builtins, *extra)
    _metrics_cache[cache_key] = result
    return result


def load_judge_metrics(project_root: Path | None = None) -> tuple[JudgeMetric, ...]:
    """Return built-in judge metrics plus project-local + plugin ones.

    Built-ins come from ``ai_eval.metrics.judge_builtin`` (the five judge
    metrics) plus the local latency pair. Project-local ``eval/metrics.yaml``
    metrics with a ``prompt_template`` are merged here. Plugins expose a
    ``JudgeMetric`` (or a compatible dataclass/dict) under the same
    ``ai_eval.metrics`` group.

    Merge order — built-ins win; project-local can only ADD; plugins fill.
    """
    cache_key = project_root
    cached = _judge_metrics_cache.get(cache_key)
    if cached is not None:
        return cached
    from ai_eval.metrics.judge_builtin import BUILTIN_JUDGE_METRICS

    builtins: tuple[JudgeMetric, ...] = (
        *BUILTIN_JUDGE_METRICS,
        *BUILTIN_JUDGE_METRICS_TUPLE,
    )
    local: tuple[JudgeMetric, ...] = ()
    if project_root is not None:
        from ai_eval.metrics.local import load_local_judge_metrics

        local = load_local_judge_metrics(project_root)
    extra: list[JudgeMetric] = []
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        result = (*builtins, *local)
        _judge_metrics_cache[cache_key] = result
        return result
    try:
        eps = entry_points(group=_EP_GROUP)
    except TypeError:
        eps = entry_points().get(_EP_GROUP, [])  # type: ignore[attr-defined]
    seen = {m.name for m in builtins}
    for m in local:
        if m.name not in seen:
            seen.add(m.name)
            extra.append(m)
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
    result = (*builtins, *extra)
    _judge_metrics_cache[cache_key] = result
    return result


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
        return JudgeMetric(name=obj, description=f"plugin judge metric {ep_name}", non_judge=False)
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


def all_names(project_root: Path | None = None) -> list[str]:
    """All registered metric names (legacy + judge + latency + local)."""
    names = {m.name for m in load_metrics(project_root)}
    names |= {m.name for m in load_judge_metrics(project_root)}
    return sorted(names)


def is_registered(name: str, project_root: Path | None = None) -> bool:
    if name in {m.name for m in load_metrics(project_root)}:
        return True
    return name in {m.name for m in load_judge_metrics(project_root)}


def get(name: str, project_root: Path | None = None) -> Metric | None:
    for m in load_metrics(project_root):
        if m.name == name:
            return m
    return None


def get_judge_metric(name: str, project_root: Path | None = None) -> JudgeMetric | None:
    """Return the judge metric for ``name`` or ``None`` if not registered.

    The run-time strict gate (:mod:`ai_eval.runner.thresholds`) calls this and
    raises :class:`MetricNotImplementedError` (exit 1) when a rubrics.yaml
    metric has no implementation — even when init-time validation only warned.
    """
    for m in load_judge_metrics(project_root):
        if m.name == name:
            return m
    return None


def is_strict() -> bool:
    return os.environ.get("AI_EVAL_STRICT_METRICS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
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
