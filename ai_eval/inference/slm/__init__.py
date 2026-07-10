"""SLM-powered rubric generation (hybrid AST + local SLM).

This package wraps a local/remote completion model (default
``ollama/qwen2.5-coder:7b``) behind :func:`complete`, which returns an
instructor-validated Pydantic object. The builder (:mod:`builder`) consumes a
:class:`~ai_eval.inference.ast_scan.ScanResult` plus per-task evidence and emits
a validated :class:`~ai_eval.config.schema.RubricsConfig`.

Importing this package does not import litellm/instructor until
:func:`complete` is first called, so ``--rubric-engine rules`` runs do not pay
the import cost.
"""

from __future__ import annotations

from .builder import RubricEngineError, build_rubrics_slm
from .client import CompleteFn, complete
from .evidence import build_task_evidence, select_entry_point_files
from .validation import validate_against_scan

__all__ = [
    "CompleteFn",
    "RubricEngineError",
    "build_rubrics_slm",
    "build_task_evidence",
    "complete",
    "select_entry_point_files",
    "validate_against_scan",
]
