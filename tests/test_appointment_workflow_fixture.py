"""End-to-end fixture test mirroring the appointment_helper repo shape.

The canonical test target (`appointment_helper/services/conversation_workflow_service.py`)
uses OpenAI Responses calls with `tools=` and a LangGraph graph split across
two methods (build in `_build_graph`, invoke in `run`). This fixture mirrors
that shape and asserts:
  - The Responses detector auto-fires a `workflow` task on `_call_model`.
  - The LangGraph detector does NOT fire (construction + invoke live in
    different enclosing defs — the deliberate false-positive guard); a hints
    file fills that gap by declaring the `run` entry explicitly.
  - With hints, both tasks survive and the repo classifies to `workflow`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.hints import merge_hints
from ai_eval.inference.synthesize import build_rubrics


def test_appointment_workflow_responses_auto_detects(
    appointment_workflow_repo: Path,
) -> None:
    """Without hints, the Responses call auto-detects a workflow task on
    _call_model. The LangGraph graph invoke (in `run`) is NOT detected
    because construction lives in a different def — the false-positive guard.

    Note: ``_call_model`` is a method inside a class, so ``find_callable_defs``
    (top-level only) returns ``entry=None`` for it — the name falls back to
    ``conversation_workflow_service_workflow``. This is the documented detector
    behavior shared across all detectors; the task still fires with the
    correct ``type=workflow``."""
    result = scan_repo(appointment_workflow_repo)
    types = {t.type for t in result.tasks}
    assert "workflow" in types
    # The Responses-driven task on _call_model is auto-detected (method, so
    # entry=None, name falls back to stem_workflow).
    responses_wf = [
        t for t in result.tasks
        if t.type == "workflow" and t.framework == "openai"
    ]
    assert len(responses_wf) == 1
    assert responses_wf[0].entry is None  # method, not top-level def
    # The graph.invoke in run() is NOT auto-detected (no construction in same def).
    run_tasks = [t for t in result.tasks if t.entry == "run"]
    assert run_tasks == []


def test_appointment_workflow_hints_fill_graph_gap(
    appointment_workflow_repo: Path,
) -> None:
    """A hints file declaring the `run` entry adds a second workflow task that
    AST missed. AST keeps its _call_model task; the hint adds `run`."""
    hints_path = appointment_workflow_repo / "eval" / "ai-evals.hints.yaml"
    hints_path.parent.mkdir(parents=True)
    hints_path.write_text(
        yaml.safe_dump({
            "tasks": [
                {
                    "name": "conversation_workflow",
                    "file_path": "services/conversation_workflow_service.py",
                    "entry": "run",
                    "type": "workflow",
                    "inputs": ["state"],
                    "outputs": ["state"],
                }
            ]
        }),
        encoding="utf-8",
    )
    scan = scan_repo(appointment_workflow_repo)
    merged = merge_hints(scan, hints_path)
    # The AST Responses task fires on _call_model (a method → entry None),
    # and the hint adds the `run` entry AST missed.
    ast_tasks = [t for t in merged.tasks if t.framework == "openai"]
    assert len(ast_tasks) == 1
    assert ast_tasks[0].type == "workflow"
    hint_tasks = [t for t in merged.tasks if t.framework == "hint"]
    assert len(hint_tasks) == 1
    assert hint_tasks[0].name == "conversation_workflow"
    assert hint_tasks[0].entry == "run"
    # No collision: the AST task has entry=None (method) while the hint has
    # entry="run", so both survive — they target different entry points.
    entries = {t.entry for t in merged.tasks}
    assert entries == {None, "run"}

    rubrics = build_rubrics(merged)
    assert rubrics.project_type == "workflow"
    assert "conversation_workflow" in rubrics.tasks
