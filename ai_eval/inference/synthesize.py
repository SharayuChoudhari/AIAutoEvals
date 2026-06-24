"""Turn a flat list of `DetectedTask`s into a `RubricsConfig` ready for disk."""

from __future__ import annotations

from collections import Counter

from ai_eval.config.defaults import (
    DEFAULT_JUDGE,
    DEFAULT_REGRESSION_JUDGE,
    SCHEMA_VERSION,
)
from ai_eval.config.schema import (
    DefaultsBlock,
    JudgeConfig,
    MetricSpec,
    ProjectType,
    RubricsConfig,
    TaskSpec,
)
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask

# Default metric set per task type. Matches design §2.6 (Phase 4 metrics).
_DEFAULT_METRICS: dict[str, list[MetricSpec]] = {
    "tool_calling": [
        MetricSpec(name="argument_accuracy", threshold=0.9, weight=1.0),
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
    "rag": [
        MetricSpec(name="context_precision", threshold=0.85, weight=1.0),
        MetricSpec(name="faithfulness", threshold=0.9, weight=1.0),
    ],
    "agent": [
        MetricSpec(name="argument_accuracy", threshold=0.85, weight=1.0),
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
    "chat": [
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
}


def _classify_project_type(tasks: list[DetectedTask]) -> ProjectType:
    if not tasks:
        return "custom"
    types = Counter(t.type for t in tasks)
    has_rag = types["rag"] > 0
    has_tools = types["tool_calling"] > 0 or types["agent"] > 0
    if has_rag and has_tools:
        return "rag_and_tools"
    if has_tools:
        return "tool_calling" if types["tool_calling"] >= types["agent"] else "agent"
    if has_rag:
        return "rag"
    return "chat"


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    counter = 2
    while f"{name}_{counter}" in used:
        counter += 1
    return f"{name}_{counter}"


def build_rubrics(
    scan: ScanResult,
    *,
    judge_default: str | None = None,
    judge_regression: str | None = None,
) -> RubricsConfig:
    """Return a `RubricsConfig` synthesized from the scan result."""
    judge = JudgeConfig(
        default=judge_default or DEFAULT_JUDGE,
        regression_check=judge_regression or DEFAULT_REGRESSION_JUDGE,
    )
    used_names: set[str] = set()
    task_specs: dict[str, TaskSpec] = {}
    for task in scan.tasks:
        name = _unique_name(task.name, used_names)
        used_names.add(name)
        task_specs[name] = TaskSpec(
            file_path=task.file_path,
            entry=task.entry,
            type=task.type,
            inputs=task.inputs,
            outputs=task.outputs,
            metrics=list(_DEFAULT_METRICS.get(task.type, _DEFAULT_METRICS["chat"])),
        )
    return RubricsConfig(
        schema_version=SCHEMA_VERSION,
        project_type=_classify_project_type(scan.tasks),
        judge=judge,
        defaults=DefaultsBlock(),
        tasks=task_specs,
    )


__all__ = ["build_rubrics"]
