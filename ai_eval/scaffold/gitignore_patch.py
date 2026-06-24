"""Append `.ai-eval/` to the project's `.gitignore` if not already present."""

from __future__ import annotations

from pathlib import Path

_BLOCK = (
    "\n# ai-eval: eval/ is checked in; .ai-eval/ is local cache\n.ai-eval/\n"
)


def ensure_gitignored(gitignore: Path) -> bool:
    """Return True if the file was modified (i.e. the line was added)."""
    if gitignore.exists():
        existing = gitignore.read_text(encoding="utf-8")
        if any(line.strip() == ".ai-eval/" for line in existing.splitlines()):
            return False
        gitignore.write_text(existing.rstrip() + _BLOCK, encoding="utf-8")
        return True
    gitignore.write_text(_BLOCK.lstrip(), encoding="utf-8")
    return True


__all__ = ["ensure_gitignored"]
