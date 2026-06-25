"""Detector abstract base class + value objects.

Each detector inspects a parsed Python module and emits zero or more
``DetectedTask`` records that ``inference.synthesize`` turns into rubric entries.

The ``extract`` signature accepts pre-computed ``calls`` and ``defs`` so that
``scan_repo`` can compute them once per file and share them across all detectors
that match — avoiding redundant ``ast.walk`` calls.

Third parties register detectors via the Python entry-point group
``ai_eval.detectors``.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ai_eval.config.schema import TaskType
from ai_eval.inference.signatures import ImportInfo


@dataclass
class DetectedTask:
    """A candidate task extracted from one module."""

    name: str
    framework: str
    type: TaskType
    file_path: str
    entry: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


class Detector(ABC):
    """Inspect one module's AST and return any task candidates it finds."""

    framework: str = "unknown"

    @abstractmethod
    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool: ...

    @abstractmethod
    def extract(
        self,
        tree: ast.AST,
        imports: list[ImportInfo],
        file_path: Path,
        project_root: Path,
        *,
        calls: list[ast.Call] | None = None,
        defs: list[ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> list[DetectedTask]: ...


__all__ = ["DetectedTask", "Detector"]
