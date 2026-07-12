"""Task selection: call-graph demotion (P2) + judge-exclusion (D1).

This is the gate between raw detector output and the rubric engine. It filters
``scan.tasks`` down to the public AI task surface by:

1. **P2 reachability demotion** — build the cross-file call graph
   (:mod:`ai_eval.inference.callgraph`) and drop any site transitively reached
   by another detected site (internal nodes, not public surfaces).
2. **Judge-exclusion** — drop sites that are judge/eval harnesses, not tasks,
   per ``J1 OR (J2 AND J3) OR J5``:
   - **J1** — the site's module imports a known judge framework (``ragas``,
     ``deepeval``, ``trulens``, ``langchain.evaluation``, ``openai.evals``,
     ``promptfoo``, ``inspect_ai``).
   - **J2** — the site's file lives under ``evaluations/``, ``eval/``,
     ``tests/``, ``benchmarks/``, ``metrics/``.
   - **J3** — the enclosing callable returns a score-shaped object (a
     ``float``/``int`` field named ``score``/``rating``/``faithfulness``/
     ``precision``, or a class whose name contains ``Evaluation``/``Metric``).
     Bare ``Result`` is NOT sufficient (would false-positive on
     ``SearchResult``).
   - **J5** — the site is inside ai-evals' own generated ``eval/tests.py``.

Both layers are escape-hatched by hints: ``judge_code: [paths]`` force-excludes
matching files; ``force_task: true`` on a hint task (matching by
``(file_path, entry)``) makes a site immune to both demotion and exclusion.

Policy is disjunctive-broad (low false-negative): silently dropping a real task
is the worst failure for a zero-config tool, so J1 alone suffices and J2 AND J3
requires both.
"""

from __future__ import annotations

import ast
import fnmatch
from pathlib import Path

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.callgraph import FileContext, build_call_graph, compute_roots
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.signatures import ImportInfo

#: Known judge/eval-framework packages (J1). A site whose enclosing module
#: imports any of these is excluded as judge code.
_JUDGE_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "ragas",
        "deepeval",
        "trulens",
        "trulens_eval",
        "langchain.evaluation",
        "openai.evals",
        "promptfoo",
        "inspect_ai",
    }
)

#: Directory prefixes that mark a file as judge/eval code (J2). Matched as
#: POSIX path prefixes against ``DetectedTask.file_path``.
_EVAL_DIR_PREFIXES: tuple[str, ...] = (
    "evaluations/",
    "eval/",
    "tests/",
    "benchmarks/",
    "metrics/",
)

#: Score-shaped field names in a return type (J3). An enclosing callable that
#: returns an object with one of these fields (or a class whose name contains
#: ``Evaluation``/``Metric``) is judge code.
_SCORE_FIELD_NAMES: frozenset[str] = frozenset(
    {"score", "rating", "faithfulness", "precision"}
)

#: Class-name substrings that mark a return type as a judge/metric object (J3).
_SCORE_CLASS_HINTS: tuple[str, ...] = ("Evaluation", "Metric")

#: ai-evals' own generated test file (J5).
_GENERATED_TESTS_PATH = "eval/tests.py"


def _matches_judge_framework(imports: list[ImportInfo]) -> bool:
    """J1: the module imports a known judge framework."""
    for info in imports:
        for fw in _JUDGE_FRAMEWORKS:
            if info.module == fw or info.module.startswith(fw + "."):
                return True
    return False


def _matches_eval_dir(file_path: str) -> bool:
    """J2: the file lives under an eval/tests/benchmarks/metrics directory."""
    p = file_path
    return any(p == prefix.rstrip("/") or p.startswith(prefix) for prefix in _EVAL_DIR_PREFIXES)


def _is_generated_test(file_path: str) -> bool:
    """J5: the site is inside ai-evals' generated ``eval/tests.py``."""
    return file_path == _GENERATED_TESTS_PATH or file_path.endswith("/" + _GENERATED_TESTS_PATH)


def _return_is_score_shaped(
    def_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """J3: the callable returns a score-shaped object.

    Inspects the function's ``return`` statements for:
      - a ``Name`` whose binding we can't resolve → not score-shaped (safe).
      - an ``Attribute`` chain ending in a score-field name (``.score`` etc).
      - a ``Call`` to a constructor whose name contains ``Evaluation``/``Metric``.
      - a ``Name``/``Call`` to a class whose name contains the hints.

    Conservative: when we can't tell, return ``False`` (do not exclude).
    """
    for node in ast.walk(def_node):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        value = node.value
        # ``return obj.score`` / ``return result.faithfulness``
        if isinstance(value, ast.Attribute):
            if value.attr in _SCORE_FIELD_NAMES:
                return True
        # ``return EvaluationResult(...)`` / ``return MetricScore(...)`` — a
        # constructor call whose name carries a hint.
        callee: ast.expr | None = None
        if isinstance(value, ast.Call):
            callee = value.func
        elif isinstance(value, ast.Name):
            callee = value
        if callee is not None:
            name = ""
            if isinstance(callee, ast.Name):
                name = callee.id
            elif isinstance(callee, ast.Attribute):
                name = callee.attr
            if any(hint in name for hint in _SCORE_CLASS_HINTS):
                return True
    return False


def _is_judge_code(
    task: DetectedTask,
    contexts: dict[str, FileContext],
) -> bool:
    """Apply ``J1 OR (J2 AND J3) OR J5`` to one task."""
    file_path = task.file_path
    if _is_generated_test(file_path):  # J5
        return True
    ctx = contexts.get(file_path)
    if ctx is not None and _matches_judge_framework(ctx.imports):  # J1
        return True
    if _matches_eval_dir(file_path):  # J2 — needs J3 to confirm
        if ctx is not None and task.entry is not None:
            def_node = ctx.entry_to_def.get(task.entry)
            if def_node is not None and _return_is_score_shaped(def_node):
                return True
    return False


def _matches_globs(file_path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(file_path, g) for g in globs)


def select_tasks(
    root: Path,
    scan: ScanResult,
    *,
    judge_code_globs: list[str] | None = None,
    force_task_keys: set[tuple[str, str | None]] | None = None,
) -> ScanResult:
    """Filter ``scan.tasks`` to the public AI task surface.

    Returns a new ``ScanResult`` whose ``tasks`` are the survivors (roots after
    demotion, minus judge code, plus force-kept tasks). ``files_scanned`` and
    ``frameworks_seen`` are preserved.

    ``judge_code_globs`` — repo-relative path globs to force-exclude (hints
    ``judge_code``). ``force_task_keys`` — ``(file_path, entry)`` pairs to
    immunize against both demotion and exclusion (hints ``force_task: true``).
    """
    judge_code_globs = judge_code_globs or []
    force_task_keys = force_task_keys or set()

    # Layer 1: call-graph demotion. Hint tasks (framework=="hint") are never
    # demoted — they're not part of the AST call graph as reachable internal
    # nodes (they're user-declared surfaces). force_task keys are also immune.
    ast_tasks = [t for t in scan.tasks if t.framework != "hint"]
    hint_tasks = [t for t in scan.tasks if t.framework == "hint"]
    edges, contexts = build_call_graph(root, scan)
    roots, reached = compute_roots(edges, ast_tasks)

    # force_task AST tasks survive even if reached.
    forced_ast = [
        t for t in reached if (t.file_path, t.entry) in force_task_keys
    ]
    survivor_ast = roots + forced_ast
    # Re-establish a stable order: by file_path then entry.
    survivor_ast.sort(key=lambda t: (t.file_path, t.entry or ""))

    # Layer 2: judge-exclusion. Apply to AST survivors; hint tasks are exempt
    # (they're user-declared, force_task flags handle override). force_task
    # keys bypass exclusion too.
    kept: list[DetectedTask] = []
    excluded: list[DetectedTask] = []
    for t in survivor_ast:
        if (t.file_path, t.entry) in force_task_keys:
            kept.append(t)
            continue
        if judge_code_globs and _matches_globs(t.file_path, judge_code_globs):
            excluded.append(t)
            continue
        if _is_judge_code(t, contexts):
            excluded.append(t)
            continue
        kept.append(t)

    # Hint tasks survive unless they match a judge_code glob (force_task hints
    # are immune even to that).
    for t in hint_tasks:
        if (t.file_path, t.entry) in force_task_keys:
            kept.append(t)
            continue
        if judge_code_globs and _matches_globs(t.file_path, judge_code_globs):
            excluded.append(t)
            continue
        kept.append(t)

    return ScanResult(
        files_scanned=scan.files_scanned,
        elapsed_seconds=scan.elapsed_seconds,
        tasks=kept,
        frameworks_seen=scan.frameworks_seen,
    )


__all__ = ["select_tasks"]
