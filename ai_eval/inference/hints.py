"""Load the opt-in hints file and merge hint tasks with AST scan results.

The hints file (``eval/ai-evals.hints.yaml``) is the escape hatch for
orchestration the AST detectors can't see (Temporal, Prefect, Airflow,
proprietary state loops) and the lever to split one detected entry point into
per-intent tasks. Hints are **additive**: when a hint and an AST task share
the same ``(file_path, entry)`` the AST task wins — hints fill gaps, they don't
override successful detection.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ai_eval.config.schema import HintsFile, HintTaskSpec
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask


def parse_hints(path: Path) -> HintsFile:
    """Parse ``ai-evals.hints.yaml`` into a validated :class:`HintsFile`.

    An absent or empty file yields an empty ``HintsFile`` (no hints). A
    malformed file raises a ``yaml.YAMLError``; a structurally-invalid one
    raises a pydantic ``ValidationError`` — both propagate to the caller so
    ``analyze``/``init`` can surface what/why/fix.
    """
    if not path.is_file():
        return HintsFile()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"hints file {path} must be a YAML mapping (got "
            f"{type(raw).__name__})"
        )
    return HintsFile.model_validate(raw)


def _hint_task_key(task: DetectedTask) -> tuple[str, str | None]:
    """Dedup key shared by hint and AST tasks: ``(file_path, entry)``.

    ``entry`` is ``None`` when no enclosing def is known (module-level code);
    two tasks with the same file_path but different entries don't collide.
    """
    return (task.file_path, task.entry)


def _hint_to_detected_task(hint: HintTaskSpec) -> DetectedTask:
    """Convert a validated hint into a ``DetectedTask`` flagged ``framework="hint"``."""
    return DetectedTask(
        name=hint.name,
        framework="hint",
        type=hint.type,
        file_path=hint.file_path,
        entry=hint.entry,
        inputs=list(hint.inputs),
        outputs=list(hint.outputs),
        evidence=[f"hint task {hint.name!r} from ai-evals.hints.yaml"],
    )


def merge_hints(scan: ScanResult, hints_path: Path) -> ScanResult:
    """Return a new :class:`ScanResult` with hint tasks merged into ``scan.tasks``.

    Merge semantics (per plan §Hints):
      - Each hint becomes a ``DetectedTask`` with ``framework="hint"``.
      - If a hint and an AST task share ``(file_path, entry)``, the AST task
        wins (hints fill gaps, they don't override successful detection).
      - Hint tasks that don't collide are appended, so both the rules and the
        SLM/hybrid engines see them via ``scan.tasks``.

    Hints are appended after AST tasks so detector order (and the AST-wins
    rule) is preserved. The returned ``ScanResult`` reuses the input's
    ``files_scanned``/``elapsed_seconds``/``frameworks_seen`` — hints don't
    add scanned files.
    """
    hints = parse_hints(hints_path)
    if not hints.tasks:
        return scan

    ast_keys = {_hint_task_key(t) for t in scan.tasks}
    merged = list(scan.tasks)
    for hint in hints.tasks:
        candidate = _hint_to_detected_task(hint)
        if _hint_task_key(candidate) in ast_keys:
            # AST already detected this entry point — keep the detector's
            # version (its evidence/type). The hint is dropped, not merged,
            # so we never emit two tasks for the same entry.
            continue
        merged.append(candidate)
    return ScanResult(
        files_scanned=scan.files_scanned,
        elapsed_seconds=scan.elapsed_seconds,
        tasks=merged,
        frameworks_seen=scan.frameworks_seen,
    )


__all__ = ["merge_hints", "parse_hints"]
