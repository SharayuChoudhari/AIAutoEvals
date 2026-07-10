"""Turn a flat list of `DetectedTask`s into a `RubricsConfig` ready for disk."""

from __future__ import annotations

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
    # Only the two metrics native to `workflow` (per the registry's
    # applicable_task_types). argument_accuracy / hallucination_rate are NOT
    # auto-assigned here — they list tool_calling/agent, not workflow, and the
    # prior plan's Risks flagged that applicability mismatch. Users add them
    # per-task when wanted.
    "workflow": [
        MetricSpec(name="task_completion", threshold=0.9, weight=1.0),
        MetricSpec(name="slot_filling_accuracy", threshold=0.9, weight=1.0),
    ],
}


def _looks_like_rag(task: DetectedTask) -> bool:
    """Defensive RAG check: an explicitly-tagged rag task, or a task whose
    inputs/outputs resemble retrieval (``query`` in inputs, ``documents`` in
    outputs) even when its ``type`` fell through to ``chat`` for an
    undetected framework. Prevents silent downgrade to ``project_type: chat``.
    """
    if task.type == "rag":
        return True
    if "documents" in task.outputs and "query" in task.inputs:
        return True
    return False


def _classify_project_type(tasks: list[DetectedTask]) -> ProjectType:
    if not tasks:
        return "custom"
    rag_count = sum(1 for t in tasks if _looks_like_rag(t))
    agent_count = sum(1 for t in tasks if t.type == "agent")
    tool_count = sum(1 for t in tasks if t.type == "tool_calling")
    workflow_count = sum(1 for t in tasks if t.type == "workflow")
    has_rag = rag_count > 0
    # Workflow tasks are tool-bearing (they drive tools through orchestration),
    # so they join tool_calling/agent in the has_tools signal.
    has_tools = tool_count > 0 or agent_count > 0 or workflow_count > 0
    if has_rag and has_tools:
        return "rag_and_tools"
    if has_tools:
        # An all-workflow repo classifies to the dedicated `workflow` literal
        # rather than collapsing into tool_calling/agent, which would mislabel
        # orchestration-heavy repos that happen to call tools. When workflow
        # tasks coexist with plain tool_calling/agent tasks, fall back to the
        # dominant tool/agent branch so mixed repos keep their existing label.
        non_workflow_tools = tool_count + agent_count
        if non_workflow_tools == 0:
            return "workflow"
        return "tool_calling" if tool_count >= agent_count else "agent"
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
        # If the project-level RAG recovery (``_looks_like_rag``) promotes this
        # task's shape to rag, keep per-task metrics consistent with the
        # promoted project_type rather than leaving a chat-typed metric set.
        task_type = "rag" if _looks_like_rag(task) and task.type != "rag" else task.type
        task_specs[name] = TaskSpec(
            file_path=task.file_path,
            entry=task.entry,
            type=task_type,
            inputs=task.inputs,
            outputs=task.outputs,
            metrics=list(_DEFAULT_METRICS.get(task_type, _DEFAULT_METRICS["chat"])),
        )
    return RubricsConfig(
        schema_version=SCHEMA_VERSION,
        project_type=_classify_project_type(scan.tasks),
        judge=judge,
        defaults=DefaultsBlock(),
        tasks=task_specs,
    )


__all__ = ["build_rubrics"]
