"""Build per-task evidence (snippets + framework hints) from a ``ScanResult``.

Evidence is what the SLM consumes. To keep prompts focused and cheap, the
snippet is the enclosing function's source (extracted via ``ast.unparse``),
truncated to ``max_snippet_chars`` — never the whole file.

The empty-evidence fallback selects entry-point files (main.py / app.py /
handlers/* / *agent* / *route*) and returns their (truncated) contents plus a
shallow repo file tree, so the SLM can propose tasks when no detector fired.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ai_eval.inference.ast_scan import iter_python_files
from ai_eval.inference.detectors.base import DetectedTask

#: Globs (POSIX, repo-relative) marking likely entry-point files. The empty
#: fallback reads up to a handful of these so the SLM has grounded context.
_ENTRY_POINT_PATTERNS: tuple[str, ...] = (
    "main.py",
    "app.py",
    "run.py",
    "server.py",
    "handlers/*.py",
    "handler/*.py",
    "routes/*.py",
    "router/*.py",
    "api/*.py",
    "*agent*.py",
    "*route*.py",
    "*bot*.py",
    "*eval*.py",
    "*infer*.py",
    "*predict*.py",
    "*classify*.py",
    "*extract*.py",
    "*summarize*.py",
)

#: Hard cap on how many entry-point files feed the empty-evidence fallback, to
#: bound the prompt size for the single repo-level SLM call.
_MAX_ENTRY_FILES = 6
_MAX_ENTRY_FILE_CHARS = 1200
_DEFAULT_MAX_SNIPPET_CHARS = 1500


@dataclass
class TaskEvidence:
    """Grounded context for one detected task."""

    task: DetectedTask
    snippet: str
    framework_hints: list[str] = field(default_factory=list)
    enclosing_function: str | None = None
    file_path: str = ""


def _function_source(
    project_root: Path, file_path: str, entry: str | None
) -> str | None:
    """Return the ``ast.unparse``-ed source of the enclosing function, or None."""
    if not entry:
        return None
    abs_path = project_root / file_path
    try:
        source = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            try:
                return ast.unparse(node)
            except Exception:
                return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_task_evidence(
    project_root: Path,
    task: DetectedTask,
    *,
    max_snippet_chars: int = _DEFAULT_MAX_SNIPPET_CHARS,
) -> TaskEvidence:
    """Build evidence for one detected task."""
    snippet = _function_source(project_root, task.file_path, task.entry)
    if snippet is None:
        # Fall back to the evidence strings the detector already emitted.
        snippet = "\n".join(task.evidence) or "(no snippet available)"
    snippet = _truncate(snippet, max_snippet_chars)

    hints: list[str] = []
    if task.framework and task.framework != "unknown":
        hints.append(task.framework)
    # The detector's own evidence lines (e.g. "openai chat.completions.create at f.py:6")
    # are strong framework signals; surface a couple.
    hints.extend(task.evidence[:2])

    return TaskEvidence(
        task=task,
        snippet=snippet,
        framework_hints=hints,
        enclosing_function=task.entry,
        file_path=task.file_path,
    )


@dataclass
class EmptyEvidence:
    """Context for the empty-evidence fallback SLM call."""

    entry_files: list[tuple[str, str]] = field(default_factory=list)  # (rel_path, content)
    file_tree: list[str] = field(default_factory=list)

    def to_prompt_fields(self) -> dict[str, str]:
        rendered: list[str] = []
        for rel, content in self.entry_files:
            rendered.append(f"### {rel}\n```python\n{content}\n```")
        return {
            "entry_files": "\n\n".join(rendered) or "(no entry-point files found)",
            "file_tree": "\n".join(self.file_tree) or "(empty)",
        }


def _matches_entry_point(rel: str) -> bool:
    from fnmatch import fnmatch

    posix = Path(rel).as_posix()
    for pat in _ENTRY_POINT_PATTERNS:
        if "/" in pat:
            if fnmatch(posix, pat):
                return True
        else:
            if Path(posix).name == pat:
                return True
            if fnmatch(Path(posix).name, pat):
                return True
    return False


def select_entry_point_files(
    project_root: Path,
    *,
    max_files: int = _MAX_ENTRY_FILES,
    max_file_chars: int = _MAX_ENTRY_FILE_CHARS,
) -> EmptyEvidence:
    """Collect entry-point file contents + a shallow repo tree for the fallback."""
    entry_files: list[tuple[str, str]] = []
    file_tree: list[str] = []

    for path in iter_python_files(project_root):
        rel = path.relative_to(project_root).as_posix()
        file_tree.append(rel)
        if len(entry_files) < max_files and _matches_entry_point(rel):
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            entry_files.append((rel, _truncate(content, max_file_chars)))

    # Sort for determinism.
    file_tree.sort()
    entry_files.sort()
    return EmptyEvidence(entry_files=entry_files, file_tree=file_tree[:200])


__all__ = [
    "EmptyEvidence",
    "TaskEvidence",
    "build_task_evidence",
    "select_entry_point_files",
]
