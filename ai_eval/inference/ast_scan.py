"""Walk a repository, parse Python files, run every detector, return findings."""

from __future__ import annotations

import ast
import fnmatch
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ai_eval.inference.detectors.base import DetectedTask, Detector
from ai_eval.inference.detectors.chromadb import ChromaDBDetector
from ai_eval.inference.detectors.langchain import LangChainDetector
from ai_eval.inference.detectors.openai_chat import OpenAIChatDetector
from ai_eval.inference.detectors.openai_tools import OpenAIToolsDetector
from ai_eval.inference.signatures import collect_imports

_DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
    "node_modules",
    ".ai-eval",
    ".kilo",
    ".agents",
)


@dataclass
class ScanResult:
    files_scanned: int
    elapsed_seconds: float
    tasks: list[DetectedTask]
    frameworks_seen: set[str]


def builtin_detectors() -> list[Detector]:
    """The Phase 1 detector set."""
    return [
        OpenAIToolsDetector(),     # before chat detector — tools take precedence
        OpenAIChatDetector(),
        LangChainDetector(),
        ChromaDBDetector(),
    ]


def load_entrypoint_detectors() -> list[Detector]:
    """Discover third-party detectors via the `ai_eval.detectors` entry-point group."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    detectors: list[Detector] = []
    try:
        eps = entry_points(group="ai_eval.detectors")
    except TypeError:  # Python <3.10 fallback shape; we require 3.10+ but be safe
        eps = entry_points().get("ai_eval.detectors", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            cls = ep.load()
            if isinstance(cls, type):
                detectors.append(cls())
        except Exception:
            # Never let a broken plugin crash a scan; surface via logs at -v.
            continue
    return detectors


def _is_ignored(rel_path: Path, extra_ignores: Iterable[str]) -> bool:
    parts = rel_path.parts
    if any(p in _DEFAULT_IGNORES for p in parts):
        return True
    rel_str = rel_path.as_posix()
    for pattern in extra_ignores:
        if fnmatch.fnmatch(rel_str, pattern):
            return True
    return False


def _is_included(rel_path: Path, include_globs: Iterable[str]) -> bool:
    if not include_globs:
        return True
    rel_str = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_str, g) for g in include_globs)


def iter_python_files(
    root: Path,
    *,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> Iterable[Path]:
    """Yield Python files in `root`, honoring default + user ignores."""
    include = tuple(include)
    exclude = tuple(exclude)
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if _is_ignored(rel, exclude):
            continue
        if not _is_included(rel, include):
            continue
        yield path


def _parse(path: Path) -> ast.AST | None:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError:
        return None


def scan_repo(
    root: Path,
    *,
    detectors: list[Detector] | None = None,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    framework_filter: Iterable[str] | None = None,
) -> ScanResult:
    """Run every detector across every Python file under `root`."""
    if detectors is None:
        combined = builtin_detectors() + load_entrypoint_detectors()
        # Dedupe by detector class so a plugin registering a builtin is harmless.
        seen_cls: set[type] = set()
        detectors = []
        for det in combined:
            if det.__class__ in seen_cls:
                continue
            seen_cls.add(det.__class__)
            detectors.append(det)
    if framework_filter:
        wanted = {f.lower() for f in framework_filter}
        detectors = [d for d in detectors if d.framework.lower() in wanted]

    started = time.monotonic()
    tasks: list[DetectedTask] = []
    frameworks_seen: set[str] = set()
    files_scanned = 0

    for path in iter_python_files(root, include=include, exclude=exclude):
        tree = _parse(path)
        if tree is None:
            continue
        files_scanned += 1
        imports = collect_imports(tree)
        for detector in detectors:
            if not detector.matches(tree, imports):
                continue
            frameworks_seen.add(detector.framework)
            tasks.extend(detector.extract(tree, imports, path, root))

    return ScanResult(
        files_scanned=files_scanned,
        elapsed_seconds=time.monotonic() - started,
        tasks=tasks,
        frameworks_seen=frameworks_seen,
    )


__all__ = [
    "ScanResult",
    "builtin_detectors",
    "iter_python_files",
    "load_entrypoint_detectors",
    "scan_repo",
]
