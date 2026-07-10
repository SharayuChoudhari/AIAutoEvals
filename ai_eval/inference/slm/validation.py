"""AST-grounded validation of an SLM-produced ``RubricsConfig``.

Guards against SLM hallucination by checking every task against the actual
scan: file paths must have been scanned, and ``entry`` (if present) must exist in
that file's top-level defs. Metrics are validated by the schema validator
(delegates to the registry). A defensive ``_looks_like_rag`` sanity check warns
(never auto-fixes) when a task's IO clearly indicates RAG but ``type`` says
``chat`` — the SLM owns the type now.
"""

from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from ai_eval.config.schema import RubricsConfig, TaskSpec
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.signatures import find_callable_defs


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_against_scan`."""

    dropped_tasks: list[str] = field(default_factory=list)
    cleared_entries: list[str] = field(default_factory=list)
    rag_misclassification_warnings: list[str] = field(default_factory=list)
    metric_warnings: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.dropped_tasks
            or self.cleared_entries
            or self.rag_misclassification_warnings
        )

    def warnings(self) -> list[str]:
        lines: list[str] = []
        for n in self.dropped_tasks:
            lines.append(f"dropped task {n!r}: file not found in scan")
        for n in self.cleared_entries:
            lines.append(f"cleared entry on task {n!r}: function not defined in file")
        for m in self.rag_misclassification_warnings:
            lines.append(m)
        for m in self.metric_warnings:
            lines.append(m)
        return lines


def _scanned_file_set(scan: ScanResult, project_root: Path) -> set[str]:
    """Return the set of repo-relative POSIX paths the scan actually parsed.

    ``ScanResult`` does not retain the file list, so re-derive it. We always
    union the detector-flagged task file_paths with a fresh repo walk: in hybrid
    mode the SLM may legitimately propose a task for a real file that the
    detectors scanned but did not flag, and such a task must not be pruned as
    hallucinated. The walk is cheap relative to the per-task SLM calls.
    """
    from ai_eval.inference.ast_scan import iter_python_files

    seen = {t.file_path for t in scan.tasks}
    for path in iter_python_files(project_root):
        try:
            seen.add(path.relative_to(project_root).as_posix())
        except ValueError:
            continue
    return seen


def _file_def_names(project_root: Path, file_path: str) -> set[str]:
    abs_path = project_root / file_path
    try:
        source = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    try:
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError:
        return set()
    return {fn.name for fn in find_callable_defs(tree)}


def _looks_like_rag(task: TaskSpec) -> bool:
    """Reuses the synthesize.py RAG heuristic but operates on a TaskSpec."""
    if task.type == "rag":
        return True
    if "documents" in task.outputs and "query" in task.inputs:
        return True
    return False


def validate_against_scan(
    rubrics: RubricsConfig,
    scan: ScanResult,
    *,
    project_root: Path,
    warn: bool = True,
) -> tuple[RubricsConfig, ValidationReport]:
    """Return a (possibly pruned) ``RubricsConfig`` and a validation report.

    Mutates ``rubrics.tasks`` in place: drops tasks whose ``file_path`` was not
    scanned, and clears ``entry`` when the named function is not defined in the
    file. Metric registry membership is enforced by the schema validator at
    construction time; here we only collect deprecation warnings.
    """
    report = ValidationReport()
    valid_files = _scanned_file_set(scan, project_root)

    for name in list(rubrics.tasks.keys()):
        task = rubrics.tasks[name]
        if task.file_path not in valid_files:
            report.dropped_tasks.append(name)
            del rubrics.tasks[name]
            continue
        if task.entry is not None:
            defs = _file_def_names(project_root, task.file_path)
            if task.entry not in defs:
                report.cleared_entries.append(name)
                task.entry = None
        # Defensive RAG sanity check — warn, do not auto-fix (SLM owns type).
        if task.type == "chat" and _looks_like_rag(task):
            msg = (
                f"task {name!r} is typed 'chat' but its IO resembles RAG "
                f"(query in inputs, documents in outputs); the SLM may have "
                f"misclassified — review before trusting"
            )
            report.rag_misclassification_warnings.append(msg)
            if warn:
                warnings.warn(msg, UserWarning, stacklevel=2)

    return rubrics, report


__all__ = ["ValidationReport", "validate_against_scan"]
