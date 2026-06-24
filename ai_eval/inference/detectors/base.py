"""Detector abstract base class + value objects.

Each detector inspects a parsed Python module and emits zero or more
`DetectedTask` records that `inference.synthesize` will turn into rubric entries.

Third parties register detectors via the Python entry-point group
`ai_eval.detectors`.
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

    name: str                       # snake_case, unique-ish; collisions resolved by synthesize
    framework: str                  # "openai", "langchain", ...
    type: TaskType                  # "tool_calling" | "rag" | "agent" | "chat"
    file_path: str                  # path relative to project root
    entry: str | None = None        # the enclosing function name, if known
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # human-readable hints


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
    ) -> list[DetectedTask]: ...


__all__ = ["DetectedTask", "Detector"]
