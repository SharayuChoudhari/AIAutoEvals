"""Best-effort git metadata for run records.

Uses ``subprocess`` git calls; returns ``git: None`` on any failure or when
the cwd isn't a repo. Never raises.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ai_eval.runner.record import GitMeta


def collect_git_meta(cwd: Path) -> GitMeta | None:
    """Return ``GitMeta`` or ``None`` if not a repo / git missing."""
    try:

        def _git(*args: str) -> str:
            r = subprocess.run(
                ["git", *args],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip() or "git failed")
            return r.stdout.strip()

        sha = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        status = _git("status", "--porcelain")
        dirty = bool(status.strip())
        return GitMeta(sha=sha, branch=branch, dirty=dirty)
    except Exception:
        return None


__all__ = ["collect_git_meta"]
