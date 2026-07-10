"""Unit tests for the `workflow` task-type defaults in synthesize + project
classification."""

from __future__ import annotations

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.synthesize import _DEFAULT_METRICS, build_rubrics


def _scan(tasks: list[DetectedTask]) -> ScanResult:
    return ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=tasks,
        frameworks_seen={t.framework for t in tasks},
    )


def test_default_metrics_workflow_has_two_native_metrics() -> None:
    metrics = _DEFAULT_METRICS["workflow"]
    names = [m.name for m in metrics]
    assert names == ["task_completion", "slot_filling_accuracy"]
    for m in metrics:
        assert m.threshold == 0.9
        assert m.weight == 1.0


def test_default_metrics_workflow_excludes_cross_applicable() -> None:
    """argument_accuracy/hallucination_rate are NOT in the workflow default
    set (they list tool_calling/agent, not workflow, per the plan's Risks)."""
    metrics = _DEFAULT_METRICS["workflow"]
    names = {m.name for m in metrics}
    assert "argument_accuracy" not in names
    assert "hallucination_rate" not in names


def test_workflow_task_gets_workflow_metrics() -> None:
    scan = _scan([
        DetectedTask(
            name="run_wf", framework="openai", type="workflow",
            file_path="wf.py", entry="run",
        )
    ])
    rubrics = build_rubrics(scan)
    spec = rubrics.tasks["run_wf"]
    assert spec.type == "workflow"
    metrics = {m.name for m in spec.metrics}
    assert metrics == {"task_completion", "slot_filling_accuracy"}


def test_all_workflow_repo_classifies_to_workflow_project_type() -> None:
    scan = _scan([
        DetectedTask(name="a", framework="openai", type="workflow", file_path="a.py", entry="a"),
        DetectedTask(name="b", framework="langgraph", type="workflow", file_path="b.py", entry="b"),
    ])
    rubrics = build_rubrics(scan)
    assert rubrics.project_type == "workflow"


def test_workflow_plus_tool_calling_classifies_to_tool_calling() -> None:
    """Mixed repo: workflow + plain tool_calling keeps the dominant tool label
    (workflow doesn't override a tool-heavy repo)."""
    scan = _scan([
        DetectedTask(
            name="wf", framework="openai", type="workflow",
            file_path="wf.py", entry="wf",
        ),
        DetectedTask(
            name="t1", framework="openai", type="tool_calling",
            file_path="t.py", entry="t1",
        ),
        DetectedTask(
            name="t2", framework="openai", type="tool_calling",
            file_path="t.py", entry="t2",
        ),
    ])
    rubrics = build_rubrics(scan)
    assert rubrics.project_type == "tool_calling"


def test_workflow_plus_rag_classifies_to_rag_and_tools() -> None:
    """Workflow is tool-bearing, so workflow+rag → rag_and_tools."""
    scan = _scan([
        DetectedTask(name="wf", framework="openai", type="workflow", file_path="wf.py", entry="wf"),
        DetectedTask(name="r", framework="langchain", type="rag", file_path="r.py", entry="r"),
    ])
    rubrics = build_rubrics(scan)
    assert rubrics.project_type == "rag_and_tools"
