"""Filesystem path resolution for the tool's working state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_eval.config.defaults import DEFAULT_EVAL_DIR, DEFAULT_STATE_DIR


@dataclass(frozen=True)
class ProjectPaths:
    """All paths the tool reads / writes for a given project root."""

    root: Path
    eval_dir: Path
    state_dir: Path

    @property
    def rubrics_yaml(self) -> Path:
        return self.eval_dir / "rubrics.yaml"

    @property
    def golden_set_json(self) -> Path:
        return self.eval_dir / "golden_set.json"

    @property
    def tests_py(self) -> Path:
        return self.eval_dir / "tests.py"

    @property
    def history_json(self) -> Path:
        return self.state_dir / "history.json"

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "config.lock"

    def ensure_state(self) -> None:
        """Create the .ai-eval/ tree if missing."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "judge").mkdir(parents=True, exist_ok=True)


def resolve_paths(
    cwd: Path,
    *,
    eval_dir: str | None = None,
    state_dir: str | None = None,
) -> ProjectPaths:
    """Compute paths from a cwd, honoring overrides for eval/ and state/."""
    eval_path = (cwd / (eval_dir or DEFAULT_EVAL_DIR)).resolve()
    state_path = (cwd / (state_dir or DEFAULT_STATE_DIR)).resolve()
    return ProjectPaths(root=cwd.resolve(), eval_dir=eval_path, state_dir=state_path)


__all__ = ["ProjectPaths", "resolve_paths"]
