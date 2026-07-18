"""Walk a repository, parse Python files, run every detector, return findings.

Performance contract:
  - Uses ``os.walk`` with in-place directory pruning so ignored directories
    (`.venv`, `node_modules`, `.git`, `__pycache__`, …) are never enumerated.
  - Computes ``calls`` and ``defs`` once per file and passes them to every
    detector, avoiding redundant ``ast.walk`` calls.

Security contract (plan §2.12):
  - Honors the project's ``.gitignore`` via ``git check-ignore --stdin`` when
    available. Falls back to the hard-coded ignore list outside a git repo or
    when git is not on PATH.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ai_eval.inference.detectors.base import DetectedTask, Detector
from ai_eval.inference.detectors.chromadb import ChromaDBDetector
from ai_eval.inference.detectors.langchain import LangChainDetector
from ai_eval.inference.detectors.langgraph import LangGraphDetector
from ai_eval.inference.detectors.openai_chat import OpenAIChatDetector
from ai_eval.inference.detectors.openai_responses import OpenAIResponsesDetector
from ai_eval.inference.detectors.openai_tools import OpenAIToolsDetector
from ai_eval.inference.detectors.pgvector import PGVectorDetector
from ai_eval.inference.signatures import (
    collect_imports,
    find_callable_defs,
    iter_calls,
)

_DEFAULT_IGNORES: frozenset[str] = frozenset(
    {
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
        ".ai-evals",
        ".kilo",
        ".agents",
        ".tox",
        ".eggs",
        "*.egg-info",
    }
)


@dataclass
class ScanResult:
    files_scanned: int
    elapsed_seconds: float
    tasks: list[DetectedTask]
    frameworks_seen: set[str]


def builtin_detectors() -> list[Detector]:
    """The Phase 1 detector set.

    Ordering encodes precedence on overlaps: tools/workflow-bearing OpenAI
    calls must be claimed before the plain-chat detector (chat skips calls
    with a tool kwarg), and Responses-with-tools emits ``workflow`` before
    chat can claim the call. LangGraph is independent (different framework
    string) but precedes LangChain in case a repo imports both families.
    """
    return [
        OpenAIResponsesDetector(),  # before chat — tools-bearing Responses → workflow
        OpenAIToolsDetector(),  # before chat — tools take precedence
        PGVectorDetector(),  # before chat — RAG takes precedence
        LangGraphDetector(),  # before LangChain — different framework, safer first
        OpenAIChatDetector(),
        LangChainDetector(),
        ChromaDBDetector(),
    ]


def load_entrypoint_detectors() -> list[Detector]:
    """Discover third-party detectors via the ``ai_eval.detectors`` entry-point group."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return []
    detectors: list[Detector] = []
    try:
        eps = entry_points(group="ai_eval.detectors")
    except TypeError:
        eps = entry_points().get("ai_eval.detectors", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            cls = ep.load()
            if isinstance(cls, type):
                detectors.append(cls())
        except Exception:
            continue
    return detectors


# ---------------------------------------------------------------------------
# .gitignore integration
# ---------------------------------------------------------------------------


def _load_gitignore_set(root: Path) -> frozenset[str] | None:
    """Return a frozenset of repo-root-relative POSIX paths that git considers
    ignored, using ``git check-ignore``.

    Returns ``None`` when outside a git repo or when git is not available.
    We enumerate all ``.py`` files that pass the hard-coded ignore filter first,
    then batch them through ``git check-ignore``. This keeps the subprocess
    invocation to one call regardless of repo size.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        git_root = Path(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # Collect all .py files to batch-check (after applying hard-coded ignores
    # only — we haven't done the gitignore pass yet).
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        _prune_dirnames(dirnames, extra_excludes=())
        for fn in filenames:
            if fn.endswith(".py"):
                candidates.append((Path(dirpath) / fn).relative_to(git_root).as_posix())

    if not candidates:
        return frozenset()

    try:
        proc = subprocess.run(
            ["git", "-C", str(git_root), "check-ignore", "--stdin", "-z"],
            input="\0".join(candidates),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # check-ignore exits 0 when at least one path matched, 1 when none matched.
    ignored = frozenset(proc.stdout.split("\0")) if proc.stdout else frozenset()
    # Normalise to paths relative to root (not git_root, which may differ).
    root_rel: set[str] = set()
    for p in ignored:
        if not p:
            continue
        try:
            root_rel.add((git_root / p).relative_to(root).as_posix())
        except ValueError:
            # Path is outside root (e.g. a parent repo file) — skip.
            pass
    return frozenset(root_rel)


# ---------------------------------------------------------------------------
# Directory walking
# ---------------------------------------------------------------------------


def _prune_dirnames(dirnames: list[str], *, extra_excludes: Iterable[str]) -> None:
    """Mutate ``dirnames`` in-place to skip ignored directories.

    This is called by ``os.walk`` before it descends — so the OS never
    enumerates inside pruned directories.
    """
    extra = tuple(extra_excludes)
    to_remove = [
        d
        for d in dirnames
        if d in _DEFAULT_IGNORES
        or any(fnmatch.fnmatch(d, pat) for pat in _DEFAULT_IGNORES if "*" in pat)
        or any(fnmatch.fnmatch(d, pat) for pat in extra)
    ]
    for d in to_remove:
        dirnames.remove(d)


def _is_user_excluded(rel_path: Path, exclude_globs: tuple[str, ...]) -> bool:
    if not exclude_globs:
        return False
    rel_str = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_str, g) for g in exclude_globs)


def _is_included(rel_path: Path, include_globs: tuple[str, ...]) -> bool:
    if not include_globs:
        return True
    rel_str = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_str, g) for g in include_globs)


def iter_python_files(
    root: Path,
    *,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    gitignored: frozenset[str] | None = None,
) -> Iterable[Path]:
    """Yield Python files in ``root`` with efficient directory pruning.

    Uses ``os.walk`` + in-place ``dirnames`` mutation so ignored directories
    are never enumerated. ``gitignored`` is a frozenset of root-relative POSIX
    paths to additionally skip (populated by ``_load_gitignore_set``).
    """
    include_t = tuple(include)
    exclude_t = tuple(exclude)
    gitignored = gitignored or frozenset()

    for dirpath, dirnames, filenames in os.walk(root):
        _prune_dirnames(dirnames, extra_excludes=exclude_t)
        dp = Path(dirpath)
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = dp / fn
            rel = path.relative_to(root)
            if _is_user_excluded(rel, exclude_t):
                continue
            if rel.as_posix() in gitignored:
                continue
            if not _is_included(rel, include_t):
                continue
            yield path


# ---------------------------------------------------------------------------
# Per-file parse + detect
# ---------------------------------------------------------------------------


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
    """Run every detector across every Python file under ``root``."""
    if detectors is None:
        combined = builtin_detectors() + load_entrypoint_detectors()
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

    # Load gitignore set once (§2.12 promise).
    gitignored = _load_gitignore_set(root)

    started = time.monotonic()
    tasks: list[DetectedTask] = []
    frameworks_seen: set[str] = set()
    files_scanned = 0

    for path in iter_python_files(root, include=include, exclude=exclude, gitignored=gitignored):
        tree = _parse(path)
        if tree is None:
            continue
        files_scanned += 1
        imports = collect_imports(tree)

        # Compute calls and defs ONCE per file — detectors receive them directly.
        calls = list(iter_calls(tree))
        defs = find_callable_defs(tree)

        for detector in detectors:
            if not detector.matches(tree, imports):
                continue
            frameworks_seen.add(detector.framework)
            tasks.extend(detector.extract(tree, imports, path, root, calls=calls, defs=defs))

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
